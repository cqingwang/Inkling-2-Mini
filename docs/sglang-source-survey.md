# Inkling-2-Mini + DSpark on 1x GB10: levers found in the serving source

Source: the `sglang` Python package of `lmsysorg/sglang:inkling-2-mini` (sglang 0.0.0.dev1+g03fde9cc3.d20261120). All paths below are relative
to that directory. FlashInfer / sgl-nvidia-mega internals were not available, so anything that depends
on them is marked unknown.

## 0. Headline (read first)

1. **Inkling2 does not use the generic MoE/GEMM runners at all.** `--moe-runner-backend`,
   `--fp4-gemm-runner-backend`, `--fp8-gemm-runner-backend` and FlashInfer autotune never reach the
   model's experts. On sm_121 (`is_sm120` = compute-capability major 12, `srt/utils/device.py:313`)
   with TP1/EP1 the experts take a hard-wired "native SM120" path: Inkling2's own routing + JIT
   quantizer + FlashInfer `group_gemm_nvfp4_nt_groupwise` (CUTLASS, W4A4) for layers 0-19 and
   FlashInfer `moe_gemm_mxfp8_nt_groupwise` (CuTe, W8A8) for layers 20-23. The tile choice is a fixed
   table tuned on an RTX 5090. No env var, server arg or autotune cache changes it.
2. **There are almost no untested target-side flags left.** The only Inkling2-specific flag that is
   off by default (`SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL`) is already on in the cookbook launch.
   Most `SGLANG_OPT_USE_INKLING_*`, `SGLANG_OPT_GATE_GEMV_MODE` and `SGLANG_OPT_LINEARIZED_SHARED_SINK`
   only affect Inkling **v1** (`srt/models/inkling.py`, `inkling_common/{moe,attn}.py`), not Inkling2.
3. **Estimated bytes moved per c1 step ≈ 12 GB**: dense BF16 attention projections 2.6 GB, target LM
   head 0.93 GB, draft vocab work ~1.8 GB, MoE experts ~6-7 GB. At ~220 GB/s that is ~55 ms against
   the ~60 ms measured. So the step is already ~90% of the bandwidth floor. Kernel tuning alone cannot
   give 2x. Losslessly, only two things can: fewer bytes per step (draft-side vocab traffic, and the
   number of distinct experts the verify reads) or more accepted tokens per step.
4. **Verify width is probably the biggest flag lever.** At c1, MoE cost grows with the number of
   *distinct* experts the 8 verify tokens touch. A single token reads 8 experts per layer visit. The
   old profile's ~28 ms of MoE fits ~27 distinct experts per visit at M=8. That growth eats most of
   the speculative gain: plain decode is roughly bandwidth-limited at ~24-32 ms per token, against
   60 ms / 2.45 = 24.5 ms per token with DSpark. `--speculative-num-draft-tokens` 5 or 6 is the
   cheapest exact arm to try. The cookbook may simply have kept the draft's training block of 8.
5. **Everything on the draft side is lossless by construction.** Draft-side changes only change what
   is proposed. Greedy verify is exact for any proposal, and sampled verify stays exact as long as the
   kept step logits are the ones sampled from. The Inkling2 draft reads the 926 MB target LM head once
   per step, plus a 201024x256 BF16 Markov fold matrix (103 MB) **7 times per step**, about 1.65 GB in
   all. That is ~7.5 ms, or 12% of the step. Only the LM-head half has a flag
   (`--speculative-dspark-lm-head-dtype mxfp8`, already queued).
6. **The ragged / adaptive verify machinery cannot cut verify width at c1** (section 4). This is
   not a flag problem: the captured graph tiers, the lagged confidence and the missing confidence
   head all rule it out.

## 1. Ranked levers

Exactness key:
- **E0** = bit-identical outputs.
- **E1** = output-exact at T=0 and exact in distribution when sampling. The target distribution is
  untouched; only the proposal or the acceptance rate changes.
- **E2** = same math, different FP32 accumulation order. Not bit-identical; T=0 can flip near-ties.
- **L** = lossy.

Rows marked "flag" can run now. Rows marked "code" need a patch.

| # | Lever | Exact setting | Expected effect and why | Exact? | Crash/OOM risk | Evidence |
|---|---|---|---|---|---|---|
| 1 | Shorter verify window (flag) | `--speculative-num-draft-tokens 6` and `5` (also `4` as a bracket). Same thing: `--speculative-dspark-block-size 5/4` | MoE bytes scale with the distinct experts across the M verify tokens, not with M. Rough model (assumptions: union ~ 8*M^0.585 fitted to the 28 ms profile, geometric acceptance a=0.6 fitted to 2.45): M=6 → ~54.5 ms, 2.38 tok/step (+7%); M=5 → ~52 ms, 2.31 (+9%). Long thinking (a≈0.72) comes out about flat. Downside: the draft was trained at block 8 and DSpark intra-block attention is bidirectional, so a shorter block may cost more acceptance than the model says. Measure both speedbench and long thinking. | E1 | Low. Draft warns on block mismatch (`dspark_config.py` resolve_runtime_config); graphs re-capture at the new width | `models/inkling2.py:376-410` (sequential loop), `inkling2_moe.py:827-911` (each routed expert's weights read once per call), `dspark_config.py:120-145` |
| 2 | Quantized draft LM head (flag, queued) | `--speculative-dspark-lm-head-dtype mxfp8` | Draft base logits read 926 MB BF16 through `torch.matmul` (cuBLAS) every step. MXFP8 is ~480 MB, so about -2 ms per step (~3%) if the kernel streams well at M=7. Target verify logits unaffected. | E1 (draft only) | **Unknown/medium.** Arg check passes on sm_121 (`is_blackwell` includes major 12, `utils/device.py:303`). The kernel is FlashInfer `mm_mxfp8(..., backend="cute-dsl")`; sm_120/121 support is not visible here and could fail at first draft forward or capture. A ~0.5 GB weight copy is cached on the lm_head (`dspark.py:72-89`). | `models/dspark.py:124-142`, `models/inkling2_mtp.py:200-207`, `arg_groups/speculative_hook.py:670-692` |
| 3 | NVIDIA megakernel probe (flag) | `--enable-nvidia-megakernel` | Swaps in a plugin forward if `sgl-nvidia-mega` registers a spec for these architectures on `sm_121`. It fails fast at argument resolution, before weights load, listing every reason it cannot run (`kernel_plugins/nvidia/binding.py:653-685`). This is a different switch from `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION`, which was screened flat. Cheapest check: `pip show -f sgl-nvidia-mega | grep -i inkling` inside the image. | unknown ("%s numerics" is logged at start) | Low: fail-fast. Speculative decoding and ragged verify must be in the spec's accepted features. | `kernel_plugins/nvidia/binding.py:529-690, 928-949` |
| 4 | Greedy-only draft (flag) | `SGLANG_DSPARK_USE_GREEDY_DRAFT=1` | Skips writing the per-step full-vocab step logits (bs x 7 x 201024 FP32 ≈ 5.6 MB per request per step) and the temperature path. The fused greedy Markov kernel does **not** apply: Inkling2's head has a softcap and a vocab limit and no `greedy_w2_weight` (`dspark_draft.py:221-237`). Expected <0.3 ms. Sampled requests lose some acceptance at T>0. | E1 at T=0. Sampled requests stay exact in law (one-hot proposal) but accept less | Low. Errors if the online estimator is on (`dspark_worker_v2.py:385-390`) | `environ.py:584-586`, `dspark_draft.py:269-310` |
| 5 | Plan prep on a side stream (flag) | `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` | DFlash-family prep kernels move to a plan stream. Only host and launch overlap; the step is GPU-bound, so expect ~0. | E0 | Low | `speculative/dflash_info_v2.py:23-35`, `spec_utils.py:1503-1510` |
| 6 | Draft vocab traffic (code) | Patch `Inkling2MTPHead` / `project_through_lm_head` | Per step: Markov fold `addmm(_fold_bias, latent, _fold_weight.T)` reads 201024x256 BF16 = 103 MB, **7 times** (~720 MB). Base logits read the 926 MB LM head via cuBLAS at M=7. Options, cheapest first: (a) send draft base logits through the sm12x GEMV (`b12x_bf16_gemv` / Triton, ≤16 rows, ~240 vs ~175 GB/s per `sm12x_bf16_gemv.py` docstring): ~-1 ms; (b) FP8 `_fold_weight` with per-row scale: ~-1.6 ms; (c) draft-only hot-vocab subset (EAGLE-style token map; `speculative_token_map` is not wired to DSpark) for both base logits and fold, with step logits = -inf outside the subset: up to ~-6 ms. Total up to ~10% of the step. | E1 (draft only; for sampling, the stored step logits must be the restricted ones) | Low; (c) needs care with the sampled-accept logits | `models/inkling2_mtp.py:46-97`, `models/dspark.py:124-142`, `dspark_draft.py:269-310` |
| 7 | Keep stacked QKVR projections on the GEMV (code) | Patch `Inkling2Attention._project_sources`, or raise `MAX_M` / `MP` in `sm12x_bf16_gemv.py` | Global layers, and every core layer in loop passes 2-3 (messages give Q/K/V separate sources), project a **3M-row** stacked input. At c1 verify, 28 of the 48 QKVR calls run at M=24 > `MAX_M=16`, so they drop from the Triton GEMV to cuBLAS on a 39.6 MB weight. Fix: split by column (q/r/gate/sinks from the q source, k from k, v from v), each at M=8 on the GEMV; same bytes, no 3x FLOPs. Est. -1 to -2 ms (bandwidth ratio 240/175 per the file docstring; cuBLAS speed at M=24 is unknown). | E2 vs today's cuBLAS on those layers. Same kernel the local layers already use | Low | `inkling2_attn.py:423-441`, `models/inkling2.py:125-153`, `kernels/ops/gemm/sm12x_bf16_gemv.py:16,90-108`, `unquant.py:1153-1159` |
| 8 | Retune NVFP4 grouped-GEMM tiles for GB10 (code) | Patch `_tiles()` in `kernels/ops/gemm/inkling2_grouped_sm120.py` | Today, tokens ≤128 → `swap_ab=True, tile_n=32, tile_k=128`, "measured best on RTX 5090 ... cold weights". GB10 has a fraction of the 5090's SMs and ~1/7 the bandwidth, so the best tile can differ. Microbench `grouped_sm120.nvfp4_expert_matmul` with real routing (tokens 8 and 64, N 2304/K 2304 and N 2304/K 1152) over tile_n {32,64,128} x tile_k {128,256} x swap_ab, report achieved GB/s. Gain unknown: 0 if already near 220 GB/s, up to several ms if not. | Probably E0 (each output is still a K-ordered FP32 MMA accumulation), but **must be checked bitwise** in the microbench; tile_k/swap_ab may reorder | Low (unsupported combinations should raise in FlashInfer; unknown) | `inkling2_grouped_sm120.py:9-18,21-42`, `inkling2_moe.py:881-911` |
| 9 | Measure the expert union first (diagnostic) | Eager monkeypatch of `Inkling2Gate.forward` to log `ids.unique().numel()` per layer visit for M=8 and M=1 | Decides between #1 and #8. If the union at M=8 is ~27, the MoE kernels run at the bandwidth limit and #1 is the lever. If the union is much lower, the MoE kernels are inefficient and #8 is the lever. Inkling2 has no routed-expert capture hook (`--enable-return-routed-experts` is not wired into `inkling2_moe.py`). | n/a | none | `inkling2_moe.py:105-139` |
| 10 | Confidence-adaptive verify width at c1 (code, large) | none exists | Needs all three: (a) graph tiers below 8 tokens at bs=1, because today compact tiers are `bs*8` only (`decode_cuda_graph_runner.py:781-784`), so `plans_verify_budget` is False at bs=1 (`dspark_planner.py:317-331`); (b) same-step confidence, because the planner uses confidence lagged 2 steps (`dspark_planner.py:1115-1128`, `SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2`); (c) a confidence source, because the Inkling2 draft has no head (`inkling2_mtp.py:130`) and would need the online estimator. Shorter verify does cut expert-union bytes, so the gain is real in principle. | E1 | High (new graph capture variants) | as cited |

### Flags that look relevant but do nothing here (or hurt). Skip them

| Flag | Why not |
|---|---|
| `--moe-runner-backend *`, `--fp4-gemm-runner-backend`, `--fp8-gemm-runner-backend`, `--speculative-moe-runner-backend` | Inkling2's experts are their own module (`Inkling2Experts`); none of these are read. The native path is chosen at `inkling2_moe.py:280-296`, the forward at `:585-589`. Marlin W4A16 (`inkling2_marlin_sm90.py`) runs only if native SM120 is impossible (EP>1 etc.). It would change numerics (no FP4 activation quant). MegaMoE and the CuTe-DSL MoE (`inkling2_mega_moe.py`, `inkling2_flashinfer_moe.py`) are SM100/103/107 only. |
| `--bf16-gemm-backend gemv` | On sm12x the hopper GEMV takes **M == 1 only** (`hopper_bf16_gemv.py:57-69`), and choosing it **turns off** the default sm12x Triton GEMV (auto-only, `unquant.py:584-597`). At M=8 that is worse. `cutedsl` raises off SM100 (`unquant.py:529-535`). |
| `SGLANG_OPT_GATE_GEMV_MODE`, `SGLANG_OPT_LINEARIZED_SHARED_SINK`, `SGLANG_OPT_USE_FUSED_GATE_TOPK`, `SGLANG_OPT_USE_INKLING_*` (multi-stream, rel-proj dispatch, fused AR/sconv/prologue/log-tau, shared fused MoE) | Read only in `srt/models/inkling_common/moe.py`, `attn.py`, `srt/models/inkling.py` (Inkling v1). Inkling2 only imports `renorm_topk_logits_scaled` from there (used only when top_k != 8). |
| `SGLANG_OPT_BF16_FP32_GEMM_ALGO` | Used only by `models/mimo_v2.py` / `kernels/ops/gemm/bf16_fp32.py`. |
| `--speculative-adaptive` (+ `_config`) | EAGLE/EAGLE3 only; for DSPARK it is switched off with a warning (`speculative/adaptive_spec_params.py:58-62`). |
| `SGLANG_RAGGED_VERIFY_MODE=cap-accept/compact`, `--speculative-dspark-sps-table-path`, `--speculative-dspark-confidence-sts-path`, `--speculative-dspark-align-verify-tokens-to-graph-tier`, `SGLANG_DSPARK_USE_ONLINE_ESTIMATOR` | See section 4. No compute saving at bs=1. cap-accept keeps the static 8-token graph and only trims acceptance. The STS path is ignored without a confidence head (`dspark_planner.py:104-143`). |
| `SGLANG_DSPARK_OPT_INT8_MARKOV_WALK` | Vanilla rank-256 head with `greedy_w2_weight` only; Inkling2's `rnn_hidden` head has none (`dspark_draft_sampler.py:328-358`). |
| `--inkling2-loop-num-passes` | Blocked with DSpark: "drafts read hidden states at fixed layer visits" (`configs/model_config.py:1164-1172`). Lossy anyway. |
| `--inkling2-loop-share-cache` | Varloop checkpoints only. Lossy (later tokens read the last pass's K/V), and gives **no** speed: each pass still reads the cache. Saves only KV memory. |
| `--speculative-use-block-verification` (queued) | Sampled drafts only; no effect on T=0 speedbench. Raises acceptance for sampled traffic; exact in law. |
| `--kv-cache-dtype mxfp8` (queued) | **Lossy**, and it **disables the compact rel-bias verify kernel** (`compact = not mxfp8 and ...`, `inkling2_attn.py:474-486`), sending verify to the padded FA4 path with a quantized Q. Helps only where global-layer KV reads dominate (long context). Label as lossy-tier only. |
| `SGLANG_ENABLE_METADATA_GLUE_GRAPH` | Forced off for the DFlash family (`decode_cuda_graph_runner.py:628-636`). |
| `SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND` | Adaptive-spec only. |
| `SGLANG_DSPARK_FP32_LM_HEAD` | Moves more bytes. |
| NV-mega sub-flags individually (`SGLANG_DSPARK_FUSED_ACCEPT_TAIL`, `SGLANG_DSPARK_DEFER_CONTEXT_KV`, `SGLANG_OPT_DSV41_DSPARK_DRAFT_PREP_FORK`, `..._POST_FOLD`, `..._ONE_LAUNCH_COMMIT_STAGE`, `..._TARGET_VERIFY_METADATA_GLUE`, `..._TARGET_METADATA_EARLY`, `SGLANG_OPT_SKIP_GLUED_REPLAY_VIEW`, `..._SAMPLING_ACCEPT_GRAPH`, `..._BLOCKWISE_REJECTION`, `SGLANG_OPT_SPEC_*`) | All default to `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` (`environ.py:607-650, 1770-1820, 2202-2235`), which was screened flat as a bundle. Host/launch work is not the bottleneck. Low priority. |

## 2. Q1: MoE kernels on sm_121

- Selection: `Inkling2Experts.native_sm120 = is_sm120 and ep_size==1 and routed kernels and can_quantize(...)`
  (`srt/models/inkling_common/inkling2_moe.py:280-296`). `forward` → `_forward_sm120` (`:585-589, 827-854`).
- Per MoE layer visit at decode/verify:
  1. `Inkling2Gate`: `linear(x, W[128,2304], fp32)`. On sm12x with ≤16 rows this is `b12x_bf16_gemv`
     (`inkling2_linear.py:237-280`; `uses_k_groups` is SM9x/10x only, `:226-234`). At c8 (64 rows) it is
     `torch.mm(out_dtype=fp32)`.
  2. `route_topk` (top-8 + bias + route_scale), then `route_metadata` (sort by expert) (`kernels/ops/moe/inkling2_route.py`).
  3. NVFP4 layers: `nvfp4_routed_quantize` (JIT CUDA `inkling2/routed_quant_sm120.cuh`), then FlashInfer
     `group_gemm_nvfp4_nt_groupwise` (CUTLASS SM120, **W4A4**), then `nvfp4_routed_swiglu_quantize`, then the second
     grouped GEMM, then `routed_partial` combine (`inkling2_moe.py:881-911`; `kernels/ops/gemm/inkling2_grouped_sm120.py:21-42`;
     `kernels/ops/moe/inkling2_experts_sm120.py`).
  4. MXFP8 layers (20-23): `mxfp8_routed_quantize`, then FlashInfer `moe_gemm_mxfp8_nt_groupwise` (CuTe,
     `scale_granularity_mnk=(1,1,32)`, no tile args), then `mxfp8_routed_swiglu_quantize`, then the second GEMM (`inkling2_moe.py:913-937`).
- Tiles: `_tiles(tokens, width)` (`inkling2_grouped_sm120.py:9-18`). tokens ≤128 gives `(swap_ab=True, tile_n=32, tile_k=128)`.
  So the same kernel config serves M=8 (c1) and M=64 (c8). This *is* the "skinny M" path: swap-AB with
  narrow N tiles. Tuned on RTX 5090, not GB10.
- Autotune: none on this path. Tiles are fixed, the JIT quantizers compile per width, and FlashInfer
  autotune is not involved. No cold cache to warm. The only first-use compiles (`b12x` projection,
  JIT quant kernels) happen before graph capture.
- Alternatives valid on sm_120/121 in this source: only Marlin (`inkling2_marlin_sm90.py`), which is
  taken only when native SM120 is impossible. It cannot be forced at TP1/EP1 by any flag, and it would
  change numerics (W4A16 instead of W4A4). No sgl-kernel/Triton fallback exists for quantized experts
  (`_lacks_routed_kernels` raises, `:590-597`).
- Does a shorter verify reduce expert bytes? Yes. The grouped GEMM walks the expert-sorted rows, so
  each expert with ≥1 routed row has its weights streamed once per call. Bytes ∝ distinct experts across
  all M·8 routing picks, which is sublinear in M. Sizes: NVFP4 expert = 4.48 MB (w13 2.65 + 0.33 scales,
  w2 1.33 + 0.17). MXFP8 expert = 8.2 MB. 44 NVFP4 and 4 MXFP8 visits per token (prelude 6 + 3x12 core +
  coda 2 NVFP4; layers 20-23 MXFP8).

## 3. Q2: BF16 dense projections at small M

| Projection | Module / path | Kernel at c1 verify (M=8) | at c8 (M=64) |
|---|---|---|---|
| qkvr (8600x2304, 39.6 MB) on local layers in pass 0 / prelude / coda | `MergedColumnParallelLinear` → `UnquantizedLinearMethod.apply` (`unquant.py:1153-1159`) | `sm12x_bf16_gemv` Triton (`kernels/ops/gemm/sm12x_bf16_gemv.py`, BN=32, BK=256, MP=16) | cuBLAS `F.linear` |
| qkvr on global layers, and all core layers in passes 2-3 | same, but **3M stacked rows** (`inkling2_attn.py:65-79, 423-441`) | **cuBLAS** (24 rows > `MAX_M=16`) | cuBLAS |
| wo_ud (2304x3072) | `RowParallelLinear` | sm12x Triton GEMV | cuBLAS |
| router gate (128x2304, FP32 out) | `inkling2_linear.linear` | `b12x_bf16_gemv` (FlashInfer b12x SIMT) | `torch.mm(out_dtype=fp32)` |
| global-layer dynamic conv `weight_proj` | `inkling2_linear.linear` (`inkling2_attn.py:356-375`) | b12x if N ≤1024, else cuBLAS | cuBLAS |
| target LM head (201024x2304, FP32 logits) | `LogitsProcessor._compute_lm_head` → `sm12x_fp32_lm_head` (`logits_processor.py:1417-1443`, `unquant.py:861-882`) | sm12x Triton GEMV, FP32 out | `torch.mm(out_dtype=fp32)` |
| draft base logits (same 926 MB weight) | `project_through_lm_head` (`models/dspark.py:124-142`) | `torch.matmul` cuBLAS, M=bs*7 | cuBLAS |
| draft Markov fold (201024x256) | `Inkling2MTPHead.step_bias` (`inkling2_mtp.py:87-97`) | `torch.addmm` cuBLAS, M=bs, **x7 per step** | same |

The sm12x GEMV is on whenever `--bf16-gemm-backend auto` (the default) is set and deterministic inference is off
(`unquant.py:584-597`). There is no split-K or tile env for it. `SGLANG_OPT_GATE_GEMV_MODE`,
`SGLANG_OPT_LINEARIZED_SHARED_SINK` and `SGLANG_OPT_BF16_FP32_GEMM_ALGO` do not reach Inkling2.
`cute_dsl ll_bf16` exists only in `layers/attention/hpc_ops_backend.py`, not on this path. Dense
weights cannot be quantized through any flag: `Inkling2QuantConfig.get_quant_method` returns None
(`layers/quantization/inkling2.py:135-136`), and Inkling2 rejects any other quant config
(`models/inkling2.py:420-424`).

## 4. Q4: verify width and adaptivity

- `--speculative-adaptive`: EAGLE/EAGLE3 only; for DSPARK it is turned off at resolution
  (`adaptive_spec_params.py:58-62`, `speculative_hook.py:272-275, 1582-1595`).
- `SGLANG_RAGGED_VERIFY_MODE` (`static` default | `cap-accept` | `compact`), `speculative/ragged_verify.py:13-28`:
  - Any non-static mode needs confidences. The Inkling2 draft has `confidence_head = None`
    (`inkling2_mtp.py:130`), so the online estimator is used (`dspark_worker_v2.py:566-582`).
  - `compact` graph tiers are `{bs*8 for captured bs}` (`decode_cuda_graph_runner.py:781-784`). At bs=1 the
    smallest tier is the full block. `plans_verify_budget` returns False and every draft is verified
    (`dspark_planner.py:317-331`, docstring says exactly this for a single request).
  - `cap-accept` keeps the static-width graph and only caps how many drafts may be accepted. No compute
    saving, and acceptance can only fall.
  - The budget uses confidence from **2 steps earlier** (`HostConfidenceBudgetPlanner._shift_to_lag`,
    `_two_steps_prior_survival`, `dspark_planner.py:1033-1128`; `SGLANG_DSPARK_CONFIDENCE_RELAY_LAG_STEPS=2`).
    It is a batch-throughput scheduler, not a per-request same-step cutoff.
- SPS table (`--speculative-dspark-sps-table-path`, built by `python -m sglang.benchmark.dspark_sps_profiler run ...`,
  format `dspark_sps.py`) and STS calibration (`--speculative-dspark-confidence-sts-path`, `sglang.benchmark.dspark_sts_fit`)
  only feed the ragged scheduler. STS is ignored without a confidence head. Neither does anything at c1.
- `--speculative-dspark-align-verify-tokens-to-graph-tier`: compact mode only; fills padding up to the tier.
  No effect at bs=1.
- **Answer:** a per-step, confidence-driven verify cut at c1 is not implementable with flags in this
  source. It would need lever #10 (new sub-block graph tiers, same-step confidence, a confidence source).
  It would be exact (only fewer drafts verified). The static version is lever #1.

## 5. Q5: reuse across loop passes / fusion / draft-verify overlap

- Loop passes are strictly sequential: pass p+1 builds its `LoopMessageBank` from pass p's blocks
  (`models/inkling2.py:376-410`). The same expert and attention weights are streamed 3x per token, and
  a layer's expert union (~100+ MB) far exceeds GB10's L2. No cross-pass weight reuse exists, and none
  is possible without changing the math.
- Per-pass KV is separate by default (`configs/inkling2.py:122-134, 185-189`). `--inkling2-loop-share-cache`
  is lossy and gives no speed.
- Draft and verify are data-dependent (the draft needs the verify's target hidden states and bonus;
  the verify needs the drafts), so there is no overlap beyond what the NV-mega bundle already does
  (deferred draft-KV store on a side stream, early verify metadata), and that bundle was screened flat.
- Existing in-graph fusions (already on): routed MoE epilogue folded into the MLP short-conv for
  decode and verify ≤32 tokens (`models/inkling2.py:61, 175-191`); fused decode prologue
  (conv + L2 clip + rel-proj) (`inkling2_attn.py:338-417`); compact rel-bias verify attention (flag on).

## 6. Q6: CPU / launch overhead

- All decode, verify and draft graphs are CUDA graphs. The NV-mega bundle (glue graphs, folded
  accept tail, side-stream draft-KV, etc.) was flat, which points to a GPU-bound step. Remaining cheap
  arm: `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` (expect ~0).
- `SGLANG_ENABLE_METADATA_GLUE_GRAPH` is force-disabled for the DFlash family. The DSpark-specific glue
  is `SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE` (part of the flat bundle).
- Kernel count per verify step is large: ~48 visits x (gate, top-k, route, 2 quant, 2 GEMM, combine,
  epilogue, attention prologue/attn/gate-sink, norms, attn-res). If an nsys trace shows inter-kernel gaps
  ≥2 µs inside the verify graph, PDL is already used by the Triton split-K/linear kernels where
  `is_arch_support_pdl()` holds. Whether the FlashInfer grouped GEMMs use PDL: unknown.

## 7. Suggested order (as written before any measurement)

1. Diagnostic #9: expert union at M=1 vs M=8 (eager hook, one short prompt), plus an nsys trace of one
   verify graph for achieved MoE GB/s.
2. Flag arms: `--speculative-num-draft-tokens 6`, `5` (and `4`); `--speculative-dspark-lm-head-dtype mxfp8`
   (queued); `--enable-nvidia-megakernel` (fail-fast probe); `SGLANG_DSPARK_USE_GREEDY_DRAFT=1`.
3. Code arms by expected value per effort: #6a/#6b (draft GEMV + FP8 fold, E1), #7 (QKVR column split,
   E2), #8 (tile microbench, check bit-equality), #6c (draft hot-vocab).
4. Honest ceiling: #1 + #2 + #6 + #7 together are optimistically ~20-25% at c1. Doubling to ~80 tok/s
   losslessly needs roughly 2x accepted tokens per step at the same step time. With no retraining,
   no lever in this source delivers that.
