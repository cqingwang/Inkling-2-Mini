// inkling_attn: PyTorch binding of the prefill attention kernel (attn_kernel.cuh).
//
// fwd(q, k_cache, v_cache, page_table, cu_seqlens_q, seqused_k, r, proj, out, lse, softmax_scale, window_left)
//   q        [T, H, 128] bf16 (last dim contiguous)
//   k_cache  [pages, page_size, H_kv, 128] bf16, v_cache the same (any strides of the first three dims)
//   page_table [B, max_pages] int32, cu_seqlens_q [n_req + 1] int32, seqused_k [>= n_req] int32
//   r        [T, H, 16] bf16, proj [16, rel_extent] bf16 contiguous
//   out      [T, H, 128] fp32 (written for the rows cu_seqlens_q covers), lse [H, T] fp32 (same rows)
//   window_left < 0: global (causal) layer; >= 0: sliding window [p - window_left, p]
// No allocation, no host sync: the launch goes to PyTorch's current stream and is CUDA-graph capturable.
// Rows of q past cu_seqlens_q[-1] are left untouched, as FA4 leaves them.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include "attn_kernel.cuh"

namespace {

using inkling_attn::Params;

bool aligned(const void* p, size_t n) { return reinterpret_cast<uintptr_t>(p) % n == 0; }

template <bool kLocal, bool kPacked, int kW>
void launch_w(const Params& p, dim3 blocks, cudaStream_t stream) {
  constexpr int smem = inkling_attn::tstage_bytes(kW);
  static bool configured[256] = {};  // per instantiation and device (the attribute is per context)
  int dev = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  TORCH_CHECK(dev >= 0 && dev < 256, "inkling_attn: device index out of range");
  if (!configured[dev]) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(inkling_attn::prefill_fwd<kLocal, kPacked, kW>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[dev] = true;
  }
  inkling_attn::prefill_fwd<kLocal, kPacked, kW><<<blocks, kW * 32, smem, stream>>>(p);
}

template <bool kLocal, bool kPacked>
void launch(const Params& p, dim3 blocks, int64_t warps, cudaStream_t stream) {
  if (warps == 8) launch_w<kLocal, kPacked, 8>(p, blocks, stream);
  else launch_w<kLocal, kPacked, 4>(p, blocks, stream);
}

void fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& page_table,
         const torch::Tensor& cu_q, const torch::Tensor& seq_k, const torch::Tensor& r, const torch::Tensor& proj,
         const torch::Tensor& out, const torch::Tensor& lse, double scale, int64_t window_left,
         const c10::optional<torch::Tensor>& projp_buf, int64_t warps) {
  TORCH_CHECK(warps == 4 || warps == 8, "inkling_attn: warps must be 4 or 8");
  TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::kBFloat16 && q.dim() == 3 && q.size(2) == inkling_attn::kD &&
                  q.stride(2) == 1,
              "inkling_attn: q must be CUDA bf16 [T, H, 128] with a contiguous last dim");
  const int64_t T = q.size(0), H = q.size(1);
  for (const auto* c : {&k, &v}) {
    TORCH_CHECK(c->is_cuda() && c->scalar_type() == at::kBFloat16 && c->dim() == 4 && c->size(3) == inkling_attn::kD &&
                    c->stride(3) == 1,
                "inkling_attn: k/v cache must be CUDA bf16 [pages, page_size, H_kv, 128] with a contiguous last dim");
    TORCH_CHECK(c->stride(0) % 8 == 0 && c->stride(1) % 8 == 0 && c->stride(2) % 8 == 0 && aligned(c->data_ptr(), 16),
                "inkling_attn: k/v cache rows must be 16-byte aligned");
  }
  TORCH_CHECK(k.sizes() == v.sizes(), "inkling_attn: k and v caches differ in shape");
  const int64_t page_size = k.size(1), hkv = k.size(2);
  TORCH_CHECK(page_size > 0 && page_size % inkling_attn::kBN == 0, "inkling_attn: page size must be a positive multiple of 64");
  TORCH_CHECK(H > 0, "inkling_attn: no heads");
  TORCH_CHECK(k.size(0) < (int64_t(1) << 31) && page_table.size(1) * page_size < (int64_t(1) << 31),
              "inkling_attn: KV cache too large");
  TORCH_CHECK(hkv > 0 && H % hkv == 0, "inkling_attn: q heads must be a multiple of kv heads");
  TORCH_CHECK(q.stride(0) % 2 == 0 && q.stride(1) % 2 == 0 && aligned(q.data_ptr(), 4), "inkling_attn: q alignment");
  TORCH_CHECK(page_table.is_cuda() && page_table.scalar_type() == at::kInt && page_table.dim() == 2 &&
                  page_table.stride(1) == 1,
              "inkling_attn: page_table must be CUDA int32 [B, max_pages] with a contiguous last dim");
  TORCH_CHECK(cu_q.is_cuda() && cu_q.scalar_type() == at::kInt && cu_q.dim() == 1 && cu_q.is_contiguous() &&
                  cu_q.numel() >= 1,
              "inkling_attn: cu_seqlens_q must be a contiguous CUDA int32 vector");
  const int64_t n_req = cu_q.numel() - 1;
  TORCH_CHECK(seq_k.is_cuda() && seq_k.scalar_type() == at::kInt && seq_k.dim() == 1 && seq_k.is_contiguous() &&
                  seq_k.numel() >= n_req,
              "inkling_attn: seqused_k must be a contiguous CUDA int32 vector with one entry a request");
  TORCH_CHECK(page_table.size(0) >= n_req, "inkling_attn: page_table has fewer rows than requests");
  TORCH_CHECK(r.is_cuda() && r.scalar_type() == at::kBFloat16 && r.dim() == 3 && r.size(0) == T && r.size(1) == H &&
                  r.size(2) == inkling_attn::kDRel && r.stride(2) == 1,
              "inkling_attn: r must be CUDA bf16 [T, H, 16] with a contiguous last dim");
  TORCH_CHECK(r.stride(0) % 2 == 0 && r.stride(1) % 2 == 0 && aligned(r.data_ptr(), 4), "inkling_attn: r alignment");
  TORCH_CHECK(proj.is_cuda() && proj.scalar_type() == at::kBFloat16 && proj.dim() == 2 &&
                  proj.size(0) == inkling_attn::kDRel && proj.is_contiguous() && proj.size(1) > 0,
              "inkling_attn: proj must be a contiguous CUDA bf16 [16, rel_extent]");
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kFloat && out.dim() == 3 && out.size(0) == T &&
                  out.size(1) == H && out.size(2) == inkling_attn::kD && out.stride(2) == 1,
              "inkling_attn: out must be CUDA fp32 [T, H, 128] with a contiguous last dim");
  TORCH_CHECK(out.stride(0) % 2 == 0 && out.stride(1) % 2 == 0 && aligned(out.data_ptr(), 8),
              "inkling_attn: out alignment");
  TORCH_CHECK(lse.is_cuda() && lse.scalar_type() == at::kFloat && lse.dim() == 2 && lse.size(0) == H &&
                  lse.size(1) == T && lse.stride(1) == 1,
              "inkling_attn: lse must be CUDA fp32 [H, T] with a contiguous last dim");
  for (const auto* t : {&k, &v, &page_table, &cu_q, &seq_k, &r, &proj, &out, &lse})
    TORCH_CHECK(t->device() == q.device(), "inkling_attn: all tensors must be on q's device");
  TORCH_CHECK(T < (int64_t(1) << 30) && n_req < (int64_t(1) << 20), "inkling_attn: too many rows");
  if (T == 0 || n_req == 0) return;

  Params p{};
  p.q = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr());
  p.q_st = q.stride(0);
  p.q_sh = q.stride(1);
  p.k = reinterpret_cast<const __nv_bfloat16*>(k.data_ptr());
  p.k_sp = k.stride(0);
  p.k_ss = k.stride(1);
  p.k_sh = k.stride(2);
  p.v = reinterpret_cast<const __nv_bfloat16*>(v.data_ptr());
  p.v_sp = v.stride(0);
  p.v_ss = v.stride(1);
  p.v_sh = v.stride(2);
  p.page_table = page_table.data_ptr<int>();
  p.pt_s = page_table.stride(0);
  p.cu_q = cu_q.data_ptr<int>();
  p.seq_k = seq_k.data_ptr<int>();
  p.n_req = static_cast<int>(n_req);
  p.r = reinterpret_cast<const __nv_bfloat16*>(r.data_ptr());
  p.r_st = r.stride(0);
  p.r_sh = r.stride(1);
  p.proj = reinterpret_cast<const __nv_bfloat16*>(proj.data_ptr());
  p.rel_extent = static_cast<int>(proj.size(1));
  p.out = out.data_ptr<float>();
  p.o_st = out.stride(0);
  p.o_sh = out.stride(1);
  p.lse = lse.data_ptr<float>();
  p.lse_sh = lse.stride(0);
  p.heads = static_cast<int>(H);
  p.group = static_cast<int>(H / hkv);
  p.page_size = static_cast<int>(page_size);
  p.total_q = static_cast<int>(T);
  p.pt_cols = static_cast<int>(page_table.size(1));
  p.num_pages = static_cast<int>(k.size(0));
  p.scale = static_cast<float>(scale);
  p.window_left = static_cast<int>(window_left);

  const int64_t bm = 16 * warps;
  const int64_t tiles = (T + bm - 1) / bm + n_req;
  const int64_t grid = tiles * H;
  TORCH_CHECK(grid < (int64_t(1) << 31), "inkling_attn: grid too large");
  const c10::cuda::CUDAGuard guard(q.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(window_left < 0 || proj.size(1) == window_left + 1,
              "inkling_attn: sliding-window layers need rel_extent == window + 1");
  const bool packed = projp_buf.has_value();
  if (packed) {
    const auto& pb = *projp_buf;
    TORCH_CHECK(pb.is_cuda() && pb.device() == q.device() && pb.scalar_type() == at::kBFloat16 && pb.is_contiguous() &&
                    pb.numel() == (proj.size(1) + 1) * inkling_attn::kDRel && aligned(pb.data_ptr(), 16),
                "inkling_attn: projp buffer must be a contiguous 16-byte aligned bf16 [(rel_extent + 1) * 16]");
    const auto pb0 = reinterpret_cast<uintptr_t>(pb.data_ptr()), pb1 = pb0 + pb.numel() * 2;
    const auto pr0 = reinterpret_cast<uintptr_t>(proj.data_ptr()), pr1 = pr0 + proj.numel() * 2;
    TORCH_CHECK(pb1 <= pr0 || pr1 <= pb0, "inkling_attn: projp buffer overlaps proj");
    p.projp = reinterpret_cast<const __nv_bfloat16*>(pb.data_ptr());
    const int n = static_cast<int>(pb.numel());
    inkling_attn::pack_proj<<<(n + 255) / 256, 256, 0, stream>>>(p.proj, reinterpret_cast<__nv_bfloat16*>(pb.data_ptr()),
                                                                 p.rel_extent);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const dim3 blocks(static_cast<unsigned>(grid));
  if (window_left >= 0) {
    if (packed) launch<true, true>(p, blocks, warps, stream);
    else launch<true, false>(p, blocks, warps, stream);
  } else {
    if (packed) launch<false, true>(p, blocks, warps, stream);
    else launch<false, false>(p, blocks, warps, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd", &fwd, "Inkling-2-Mini prefill attention with the relative bias computed in-kernel",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("page_table"), py::arg("cu_q"), py::arg("seq_k"),
        py::arg("r"), py::arg("proj"), py::arg("out"), py::arg("lse"), py::arg("scale"), py::arg("window_left"),
        py::arg("projp_buf") = py::none(), py::arg("warps") = 4);
}
