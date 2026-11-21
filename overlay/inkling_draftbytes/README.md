# inkling_draftbytes: fewer bytes per step on the DSpark draft (Inkling-2-Mini, 1x GB10)

An import-time overlay for `lmsysorg/sglang:inkling-2-mini` (sglang
0.0.0.dev1+g03fde9cc3.d20261120). It changes **only what the Inkling2 MTP draft
proposes**. The target forward, the verify and the accept rule are not
touched. Nothing here has run on a GPU yet. The CPU tests pass (23/23).
`gpu_check.py` and `bench_draft.py` run inside the image on a Spark with the server stopped (copies in overlay/tools/).

## Options (`INKLING_DRAFT_BYTES`, comma list; unset or empty = no-op)

| option | what it does | bytes/step at c1 (stock -> new) | expected | extra memory |
|---|---|---|---|---|
| `gemv` | Draft base logits (926 MB BF16 LM head, M = 7*bs rows) go through the image's `sm12x_bf16_gemv` Triton GEMV when M <= 16, instead of cuBLAS `torch.matmul`. `INKLING_DRAFT_GEMV_MAX_M` (default 16) also sends 16 < M <= that value to this package's unscaled kernel. | same bytes, better streaming | -0 to -1.4 ms (the image's docstring: ~175 GB/s for cuBLAS vs ~240 for the GEMV at 1 row; M=7 is unmeasured) | 0 |
| `foldk` | The Markov fold `addmm(fold_bias, latent, fold_weight.T)` through this package's kernel on the **BF16** fold matrix. This is a pure kernel A/B against cuBLAS at M = bs. | same | unknown (cuBLAS at M=1, K=256, N=201024) | 0 |
| `fold8[:row\|gN\|mx]` | FP8 e4m3 copy of the 201024x256 fold matrix, with a per-row fp32 scale by default. It feeds the same kernel (dequant on the fly, FP32 accumulate). | 7 x 103 MB -> 7 x 52 MB (-354 MB) | -1.6 ms | +52 MB |
| `head8[:row\|gN\|mx]` | FP8 copy of the LM head for the draft base logits, own kernel. It is the fallback if `--speculative-dspark-lm-head-dtype mxfp8` (FlashInfer cute-dsl) does not run on sm_121. It stands down when that flag is set. | 926 -> 464 MB | -2.1 ms | +464 MB |
| `draftmx8[:mx\|row]` | MXFP8 (default) copies of every BF16 linear in the 3 draft layers (qkv 5120x2304, o 2304x3072, gate_up 18432x2304, down 2304x9216, kv_proj_ctx 2048x2304), made in memory after load. Rows <= `INKLING_DRAFT_W8_MAX_M` (64) use the FP8 kernel. Larger batches use the stock BF16 path. | ~524 -> ~270 MB | -1.1 ms | +270 MB (BF16 kept) |
| `hotvocab:N` | The draft proposes only from the first N ids of `INKLING_DRAFT_HOTVOCAB_FILE`, which must be frequency-ranked (see `build_hotvocab.py`). The base logits read N LM-head rows, and the fold reads N fold rows. Stacks with `fold8`, `head8` and `gemv`. | N=32768: 1647 -> ~210 MB with fold8 (~135 MB with head8 too) | -6.3 to -6.9 ms, minus the acceptance lost to misses | +151 MB (N=32K, BF16 head rows) + 8-17 MB |

Byte math at 220 GB/s: LM head 201024x2304x2 = 926 MB; fold 201024x256x2 = 103 MB,
read 7 times = 720 MB; draft linears 165 MB per layer x 3. The ms figures are the
bytes saved / 220 GB/s. They are upper bounds that hold only if the new kernels
stream at the stock rate. `bench_draft.py` measures the actual rate.

Stacked best case on paper: `hotvocab:32768,fold8,head8,draftmx8` saves ~8 ms of
~60 ms (~13 %) before acceptance changes. `gemv,fold8,draftmx8` gives -3 to -4 ms
with no change to the proposal vocabulary. Acceptance must be measured.

Tunables (env): `INKLING_DRAFT_FOLD_BN` (64), `INKLING_DRAFT_FOLD_BK` (128),
`INKLING_DRAFT_LINEAR_BN` (32), `INKLING_DRAFT_LINEAR_BK` (256),
`INKLING_DRAFT_NUM_WARPS` (4). Set them from the `bench_draft.py` sweeps.

## Patch points

All patches are installed by post-import hooks when the target module is first
imported. Weights are prepared once, right after
`Inkling2ForConditionalGenerationMTP.attach_shared_modules` (draft worker init),
which is before any CUDA graph capture.

| target (file:function) | option | change |
|---|---|---|
| `srt/models/inkling2_mtp.py: Inkling2ForConditionalGenerationMTP.attach_shared_modules` | any | runs the stock attach first, then `prepare()`: FP8 packs, hot subsets, draft-linear packs, and one log line |
| `srt/models/inkling2_mtp.py: project_through_lm_head` (the name this module imported from `models/dspark.py`, so only the Inkling2 draft is affected) | gemv, head8 | GEMV / FP8 kernel. It stands down if the LM head is quantized or `--speculative-dspark-lm-head-dtype` is set |
| `srt/models/inkling2_mtp.py: Inkling2MTPHead.step_bias` | fold8, foldk | same `_rnn_step`, then the fold kernel instead of `torch.addmm`, then the same `_gather_logits` crop |
| `srt/models/inkling2_mtp.py: Inkling2MTPHead.refresh_fold` | any | after a weight refresh, re-prepares the packs in place, so captured graphs keep valid pointers |
| `srt/models/inkling2_mtp.py: Inkling2ForConditionalGenerationMTP.compute_base_logits` | hotvocab | base logits on the N hot rows, scattered into a full-width [M, V] buffer that is -inf elsewhere |
| `srt/speculative/dspark_components/dspark_draft.py: _draft_step` | hotvocab | if the head has a hot state: `_rnn_step`, then one fused kernel (fold on the hot rows + base + the stock roundings + softcap + vocab limit, scattered into a -inf row), then the **stock** `MarkovSampleStep` with a zero bias and no softcap. It writes the kept step logits and picks the token. Otherwise it calls the stock step |
| draft linears' `quant_method` (instances under `model.layers`) | draftmx8 | swapped for a subclass of the same class, so `isinstance` checks still pass, whose `apply` uses the FP8 kernel at <= 64 rows |

The fail-loud rules:
- A missing patch target raises `DraftBytesPatchError` at import. It derives from
  `BaseException` because sglang's model registry swallows `Exception`.
- A bad option string exits the process with code 78 from `sitecustomize.py`.
  Python ignores sitecustomize exceptions, so raising would not stop it.
- `prepare()` raises if TP > 1, if the LM head is not BF16, or if the
  `_draft_step` patch did not apply.

There is one startup line: `[inkling_draftbytes] active: options=...; patched ...; <packs and sizes>`.

Not compatible with `--enable-nvidia-megakernel`: its specialized draft sampler
would bypass the patched step. It was screened flat anyway.

## Exactness

Every option changes only the draft's proposal q. Nothing touches the target
distribution p or how a token is accepted.

- **Greedy (T=0) verify** accepts a drafted token iff it equals the target argmax.
  It then emits the target's own token at the first mismatch. The emitted
  sequence is therefore the target's greedy sequence for **any** proposal. All
  options are exact by construction: FP8 noise, a different kernel, a restricted
  vocabulary.
- **Sampled verify** (speculative sampling: accept x ~ q with min(1, p(x)/q(x)),
  otherwise resample from norm(max(0, p - q))) returns exactly p for **any** q, as
  long as the q the accept rule reads is the q the token was drawn from. In this
  code, `MarkovSampleStep` writes the step logits it selects from (Gumbel-max at
  the row temperature) into `logits_out`. The accept path then turns exactly those
  logits into q (`SoftmaxTemp`, or block-wise rejection). The patches keep that
  pairing:
  - gemv / foldk / fold8 / head8 / draftmx8 only change the numbers that flow
    into `MarkovSampleStep`. It stores and samples from the same values.
  - hotvocab: the fused kernel writes the final hot step logits into a full-width
    row that is -inf off the hot set. The stock `MarkovSampleStep` then runs with a
    zero bias (`x + 0`, and the BF16 round of a BF16 value is the identity) and no
    softcap, which was already applied in the hot kernel with the stock rounding
    order. So the kept logits are exactly the sampled distribution, with q = 0 off
    the set. Off-set tokens are never proposed, and the residual max(0, p - q) = p
    there gives them their full target mass. This is checked in closed form in
    `tests/test_draftbytes.py::RestrictedProposalExactness`. -inf already occurs
    in these logits (the stock vocab limit), so the accept code handles it.
  - Why not put -inf into the base logits and keep the stock step? The stock step
    applies softcap after the add, and tanh(-inf) * cap = -cap. That is a finite
    logit, so off-set tokens would keep a small probability. It would still be
    exact (q is still what is sampled from), but the proposal would be worse.
    The patched step avoids that.
- Cost is acceptance only. FP8 copies perturb q (fold8 rel. error ~2.3 %, see
  below). hotvocab rejects every position whose target token is outside the set.
  About 1 % of misses costs about 1 % of accepted drafts.

## Why the 7 fold reads cannot be fused

Step s needs `latent_s = RNN(state_{s-1}, embed(token_{s-1}), h_s, f)`, and
`token_{s-1}` is sampled from the full-vocab logits of step s-1, which include
that step's fold. The chain is strictly sequential, so no step's fold can be
computed before the previous token exists. Algebra does not help either:
`fold = lm_head @ hidden_out / mult`, so `base + bias = lm_head @ (h_s +
hidden_out(latent_s)) / mult`. Folding the bias into one pass would read the
926 MB head 7 times instead of the 103 MB fold 7 times. The rank-256 fold is
already the cheapest form. The remaining levers are fewer bytes per read (FP8)
and fewer rows (hotvocab). With hotvocab plus fold8 at N = 32768, the fold is
8.4 MB, and it may stay in L2 across the 7 steps (GB10 L2 size not confirmed
here). The rnn_proj weight (7.9 MB, an `nn.Linear`) is also read 7 times per
step (~55 MB). It is left alone.

## Numerics (CPU reference, fold-shaped data: 8192 x 256, 8 latent rows)

Relative L2 error of the FP8 fold vs the BF16 fold:

| mode | rel. error | test bound |
|---|---|---|
| row | 0.0233 | 0.035 |
| g32 | 0.0215 | 0.035 |
| g64 | 0.0224 | - |
| mx | 0.0239 | 0.040 |

e4m3 keeps 3 mantissa bits, so per-element rounding is at most 2^-4 relative.
Finer scale groups barely help because the error comes from the mantissa, not
from range. Per-row is the default (fewest scale bytes).

Kernel math: FP8 to BF16 is exact, and e8m0 scales are powers of two, so the MX
dequant in BF16 is exact. fp32 scales multiply the FP32 partial of each K tile.
The kernel therefore equals `quant.linear_ref` up to FP32 summation order, and
`gpu_check.py` gates within 1 BF16 ulp on >= 99.9 % of outputs.

## Docker additions

```
-v /path/to/repo/overlay:/opt/inkling-overlay:ro
-e PYTHONPATH=/opt/inkling-overlay          # append to any PYTHONPATH the image already sets
-e INKLING_DRAFT_BYTES=gemv,fold8          # for example
-e INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/hot.json   # only with hotvocab:N
```

Check `docker inspect lmsysorg/sglang:inkling-2-mini | grep PYTHONPATH` first.
`-e PYTHONPATH=` replaces the image's value. `sitecustomize.py` arms the overlay
in every interpreter, including the spawned scheduler. It is silent without the
env, and it chains to any other `sitecustomize.py` it shadows. A combined loader
only needs `import inkling_draftbytes; inkling_draftbytes.install()`. It is
idempotent and does not import sglang.

## Test plan

1. **CPU (done here):** `python -m unittest discover -s tests -v`. It covers
   options, the no-op contract, FP8 numerics vs BF16, the patch plumbing end to
   end against CPU stand-ins of the patched modules (`tests/stubs`), and the loud
   failures. With the hot set equal to the whole valid vocab, it checks that the
   hot path is bit-identical to stock (tokens and kept logits).
2. **GPU check (server stopped, ~3-5 GB GPU, nothing pinned):**
   `python3 /opt/inkling-overlay/gpu_check.py` (`--vocab 65536` for a smaller run).
   - Part 1: kernels vs reference (gated).
   - Part 2: the real `Inkling2MTPHead`, real `draft_markov_block` and
     `MarkovSampleStep`, run for every variant against stock. Hot-step structure
     is gated. Each variant is captured in a CUDA graph, and its replay must
     equal eager bit for bit (gated).
   - Part 3: ms of the draft vocab work per step for each variant.
   - Exit 0 = pass.
3. **Microbench:** `python3 /opt/inkling-overlay/bench_draft.py --json /tmp/b.json`.
   It times stock vs ours for the LM head (M 1..56, plus FlashInfer `mm_mxfp8`
   if it runs on sm_121), the fold (M 1..8, with BN/BK/warps sweeps), hot vocab
   per step, and the draft linears (with BN/BK sweeps). Set the tunables from it.
   If `head8`/`draftmx8` stream below ~200 GB/s at M=7/8, the tiles need work
   before a serving arm.
4. **Hot list:** collect output traces on the Spark (SGLang `/generate` with
   `output_ids` is best, since it includes special tokens exactly as drafted, or
   OpenAI responses including `reasoning_content`). Then run
   `python3 build_hotvocab.py --tokenizer /model --out hot.json traces/*.jsonl`
   and read the held-out coverage at 8K/16K/32K/64K. Pick N where held-out
   coverage is >= ~99 %.
5. **Serving arms** (one boot each, same workload, record speedbench c1/c4 tok/s
   and mean accept length from the server log):
   - `stock`, `gemv`, `fold8`, `gemv,fold8`, `draftmx8`, `gemv,fold8,draftmx8`
   - `head8` only if the mxfp8 flag fails
   - `hotvocab:N,fold8` (+`gemv`, +`head8`)

   Gate: T=0 outputs on the eval set equal stock (target path untouched). If any
   differ, bracket them with a stock-vs-stock rerun before blaming the overlay.
   Speed must beat stock beyond the noise band, and the accept-length drop must
   not eat the step-time gain.

## Files

- `inkling_draftbytes/`
  - `__init__.py`: `install()`
  - `_opts.py`: env parsing
  - `_hooks.py`: post-import hooks
  - `patches.py`: patches and `prepare()`
  - `ops.py`: CUDA/Triton vs torch-reference dispatch
  - `_triton.py`: the `w8_mm` and `hot_step` kernels
  - `quant.py`: FP8 packs and reference math
  - `timing.py`: graph-replay timing with L2 flush
- `sitecustomize.py`: single-track loader
- `gpu_check.py`, `bench_draft.py`: GPU checks and microbench (run in the container)
- `build_hotvocab.py`: frequency-ranked hot list and coverage from traces
- `tests/`: CPU unit tests and CPU stand-ins of the patched sglang modules

## Known unknowns

- Triton kernel speed on GB10 at these shapes (fp8 to bf16 convert plus `tl.dot`
  at MP=16) has not been measured. `bench_draft.py` decides.
- cuBLAS rate for the stock draft LM head at M=7 and the stock fold at M=1 is
  unknown, so the `gemv`/`foldk` gains are unknown until benched.
- GB10 L2 size, which decides whether the hot fold stays resident across the 7
  steps.
- The acceptance cost of fold8/head8/draftmx8 noise and of hot-set misses. Only a
  serving run measures it.
