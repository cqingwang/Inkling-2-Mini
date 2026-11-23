# Inkling-2-Mini on one NVIDIA DGX Spark

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark
(GB10, sm_121, 121.6 GiB unified memory) with the model's own NVFP4 / MXFP8 weights and its bundled DSpark draft,
on SGLang, plus a set of import-time speed overlays that leave the model's outputs as they are.

**Status (2026-11-23, v2.7):** one profile. SGLang `lmsysorg/sglang:inkling-2-mini` with the cookbook's DGX Spark DSpark
settings, and fourteen overlay packages (`overlay/`) that cut the bytes read per decode step (weights, draft and the
global-layer KV at long context), the kernel time of the concurrent verify batches and of the sampled accept, and the
work per prefill chunk. Against the stock cookbook cell, speedbench is +59 % at c1, concurrent prose decode is
+28 to +43 % at c4-c16, thinking load is +21 to +37 %, cold prefill is +34 to +61 %, and decode at 128K is +25 %.
The default KV pools hold about 1.59M global-layer tokens; the opt-in packed KV store (`KV12S=1`, see "Packed global
KV store") holds 2.17M. The target distribution is unchanged. v2.7 adds the certified head for 17-192-row verify
batches, a long-prefix prefill fix (1.4-1.5x per global layer at 512K-1M prefixes) and a measured 512K / 1M-context
option. The speed tables below are the v2.6 full runs; a v2.7 quick check of the default build matches them (greedy
texts identical to v2.6 on 32/32 speedbench quick prompts), and full v2.7 runs follow.

| | stock SGLang cookbook cell (mem 0.62) | this recipe |
|---|---:|---:|
| speedbench full, all 73 prompts, c1 | 40.6 | **64.4** |
| decode, prose, aggregate tok/s at c4 / c8 / c16 | 75.2 / 117.0 / 186.2 | **107.5 / 161.8 / 239.1** |
| thinking load c1 / c8 / c24 (effort high, T=0.8) | 52.3 / 193.6 / 374.9 | **71.5 / 236.4 / 451.9** |
| cold prefill, 8K / 128K tokens, tok/s | 3,399 / 2,422 | **5,478 / 3,805** |
| decode at 128K context, c1, tok/s | 37.1 | **46.4** |
| KV pool (global-layer tokens) / MemAvailable after start | 843K / ~40 GiB | **1.59M / 6.3 GiB** (`KV12S=1`: 2.17M / 6.9 GiB) |

## Speed

Measured 2026-11-23 on one Spark per run, GPU clock capped at 2200 MHz by the host's own timer, fresh start of the
container, MemAvailable logged every 5 s. The full benchmarks used `MEM_FRACTION=0.94` (`results/V26-s05/`,
`results/V26-s06/`). Speed tables show the median of the two boxes. Throughput was flat across the memory sweep.
The memory figures for the release defaults use `MEM_FRACTION=0.93`. The stock engine was measured on both boxes
with the same harness (`results/F1-sglang-base-s05/`, `results/F2-sglang-base-s06/`).

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories,
c1, temperature 0, 256 output tokens, reasoning effort low (the model always thinks; its reasoning tokens count as
decode tokens). Decode tok/s, geometric mean per category:

| prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---:|---:|---:|---:|---:|---:|---:|
| **54.1** | 72.8 | 68.6 | 77.5 | 44.1 | 75.9 | **64.4** |

Stock engine on the same boxes: prose 32.6, code 48.0, structured 44.3, agentic 49.3, multilingual 25.3, long
context 51.0, all 40.6. Spark_05 / Spark_06 scored 64.46 / 64.35 (stock 41.09 / 40.16). All 73 prompts were valid
in both runs. Median time to first token was 158.8-159.4 ms (stock 184 ms), with 2.383 tokens per verify step
(stock 2.45) at 37.2 ms per step (stock 60.9 ms).

### Decode by prompt type and concurrency

`bench/decode_table.py`: speedbench's prompts split by type (prose, code, structured, JSON), temperature 0, 256 new
tokens, reasoning effort none (the model still writes a short thought). c1 is the decode speed of one stream (median
of three requests); above c1 the aggregate over the level's wall time, per-stream decode in brackets.

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **53.0** | 72.7 (37.5) | 107.5 (28.6) | 161.8 (21.2) | 239.1 (16.1) |
| code | 62.8 | 87.3 (45.3) | 122.3 (34.0) | 177.3 (25.0) | 264.8 (18.2) |
| structured | 70.5 | 93.8 (49.0) | 128.2 (33.7) | 177.1 (24.6) | 264.3 (18.0) |
| json | 62.5 | 88.3 (46.7) | 132.1 (35.9) | 183.6 (23.9) | 280.5 (19.7) |

Stock engine, same protocol: prose 33.2 / 51.0 / 75.2 / 117.0 / 186.2, code 43.6 / 66.8 / 87.2 / 134.8 / 221.3,
structured 49.7 / 66.4 / 89.9 / 141.8 / 225.1, json 46.5 / 58.8 / 89.3 / 141.9 / 232.2. At one or two streams the
gain comes from verifying fewer draft rows and reading fewer bytes per row (the draft proposes from a 64K-token vocabulary); from four streams up it comes from the
batch-aware verify policy (the expert union of 8 to 16 concurrent requests is not saturated), the verify attention
kernel at every batch size, the packed GEMM for the 17-256-row projections, GB10-tuned expert GEMM tiles and the
draft statistics computed in one read of the draft logits.

### Thinking workloads

`bench/think_load.py`: seeded arithmetic puzzles, reasoning effort high, temperature 0.8, 2,048 (c1) or 4,096 tokens
per request, aggregate completion tok/s (median per-request decode in brackets). This is the shape of an agent
workload: long reasoning, many requests at once.

| c1 | c8 | c24 |
|---:|---:|---:|
| **71.5** (72.0) | 236.4 (34.0) | 451.9 (21.7) |

Stock engine: c1 52.3, c8 193.6, c24 374.9. One run per box per cell (sampled); c8 and c24 vary ~5 % from run to run.
Sampled requests accept their drafts by block verification under the verify cap, with the accept computed straight
from the logits from three requests up.

### Prefill and long context

speedbench `longctx`: a generated document with a unique salt (no prefix-cache hit) plus a question; cold TTFT,
then the same document with another question (served from the prefix cache); decode at 32K context.

| | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| prompt tokens | 1,008 | 8,151.5 | 32,703.5 | 130,922 |
| TTFT cold | 0.35 s | 1.49 s | 6.37 s | 34.4 s |
| **prefill, cold, tok/s** | **2,909** | **5,478** | **5,136** | **3,805** |
| TTFT, document cached | 0.22 s | 0.22 s | 0.22 s | 17.3 s |
| decode tok/s at that context | - | - | 71.0 | 46.4 |

Stock engine: 2,163 / 3,399 / 3,232 / 2,422 tok/s cold (TTFT 0.47 / 2.40 / 10.1 / 54.0 s), decode 55.1 at 32K and
37.1 at 128K. The 128K cached TTFT is the median of one cache hit (~0.3 s) and one miss (the second 128K document
evicts sliding-window state of the first), the same as the stock engine. Decode at long context is dominated by the
global layers reading the whole context (4.3 GB of BF16 KV per step at 128K, at the memory-bandwidth floor): here the
committed pages are read in an exactly coded 12-bit form (1.33x fewer bytes). The draft's global layer reads the
whole context. Decode in the two-document samples varies with the tokens per step of each generated text.

At 200K / 250K tokens (`results/V26-s05/K/`, `results/V26-s06/K/`), cold prefill was 3,141-3,159 / 2,688-2,721
tok/s (TTFT 65.1-65.4 / 93.8-94.9 s), and c1 decode was 40.6-43.3 / 34.9-35.2 tok/s.

### Memory and start-up

| `MEM_FRACTION=0.93`, `SWA_RATIO=0.05` | Spark_05 / Spark_06 |
|---|---|
| MemAvailable after start | 5.8 / 6.1 GiB |
| lowest MemAvailable with thinking c24 (48 x 2,048 tokens) alongside a cold 240,104-token needle prefill | 5.2 / 5.4 GiB, thinking 299.9 / 298.6 tok/s, needle PASS on both |
| global-layer KV pool | 1,551,104 / 1,557,120 tokens |
| warm start-up to the first answer | 131 / 137 s (137 / 121 s in the full benchmarks at 0.94) |

Details in "Memory budget" below.

### Quality checks

Same runs, same checks as for the stock engine (`bench/full.sh`):

- speedbench quality guard (30 exact-answer items, T=0): **30 / 30** on both boxes (stock engine: 30 / 30).
- The draft vocabulary lever changes proposals only: in the same-day A/B the 30 greedy speedbench texts were byte
  identical with and without it on both boxes.
- `bench/sanity.py`: 11 / 11 on both boxes: two chats, a reasoning puzzle at effort none / low / medium / high
  (all correct; effort reaches the template, top-level `reasoning_effort` and `chat_template_kwargs` both work),
  a tool call and the answer after the tool result, a JSON-schema constrained answer.
- Both boxes passed the needle with the passphrase at 50 % depth, returning `MEADOW-FALCON-0347` at 120,097
  tokens and `WILLOW-FALCON-3624` at 240,104 tokens.
- What "exact" means here: no lever changes the model's weights or its output distribution. The verify cap only
  decides how many draft tokens to verify (greedy output is still the target's argmax; for sampled requests the
  width is a left-to-right stopping rule on the draft's own probabilities, Monte Carlo tested); sampled requests are
  accepted by block verification, whose output law is the target's (Sun et al.; under the cap the dead drafts enter
  as a stopped draft, Monte Carlo tested), with the accept probabilities computed from the logits to FP32 rounding;
  the draft levers only change proposals; the packed 12-bit KV decodes to the same BF16 bits and gives attention
  partials bit-identical to stock; the certified LM head returns the exact argmax of the full head (proved per step, 0
  mismatches in a 45,584-row serving check); the kernel levers compute the same math, bit-identical where stated,
  otherwise with a different floating-point summation order (packed GEMVs/GEMMs whose weights decode to the same
  BF16 bits, the fused prefill kernels; each checked against an FP32/FP64 reference to be no less accurate than stock). Any
  such change moves near-tie tokens on this model (its W4A4 activation quantisers amplify last-bit differences;
  equally correct implementations agree on ~95 % of tokens teacher-forced), so greedy texts are not byte-identical
  to the stock engine; the stock engine itself is deterministic from boot to boot.

## What the overlays do

Decode at c1 is bound by memory bandwidth: a step reads ~13.4 GB (BF16 dense projections 2.8 GB over 48 layer
applications, LM head 0.93 GB, draft ~1.65 GB, and the routed experts touched by the 8 verify rows, ~6.5 GB) and
the stock engine runs at 1.08x the 240 GB/s floor with the GPU 99 % busy ([profile](docs/profile/PROFILE.md)).
So the overlays cut bytes, not overheads. With several requests in flight the step is still mostly bytes (c8: 117 ms,
of which 76 ms routed experts), but kernels built for one row lose time there: cuBLAS streams the dense projections at
150-200 GB/s, and above 32 verify rows the attention left the split-KV kernel for padded FA4 with a sheared bias.
Sampled requests add full-vocabulary FP32 probability tensors for the accept (~1 GB and ~5 ms per step at c24,
T=0.8). At long context the global layers' BF16 KV dominates the step (4.3 GB at 128K, plus 0.54 GB for the draft's
global layer). Prefill is the opposite: the kernels run near their peaks, but intermediates make round trips through memory (FP32
[T, 2304] activations between the loop-message mixes, norms and residual updates), the relative position bias was
materialised as a [t, 24, 1024] tensor, and the stacked Q/K/V projection computed three times the rows it keeps
([prefill analysis](docs/prefill/PREFILL.md), [round-2 profiles](docs/experiments.md#round-2-v2)).

The levers are switched by environment variables of `scripts/launch.sh`. The overlays are import-time patches in
`overlay/` (`OVERLAY=0` disables them). Effects are from screens against same-box, same-day controls
(speedbench quick: 30 prompts x 160 tokens, c1, T=0; decode table; thinking
load; at long context the step time = tokens per step / decode tok/s); the full stack is confirmed in the
tables above. Raw files: `results/screen/`, `results/screen-r2/`; details and rejected levers: [docs/experiments.md](docs/experiments.md).

| lever | switch (default) | effect, alone | output | credit |
|---|---|---|---|---|
| Verify-width policy with dead-row expert remap: per request and step, verify only the first L of the 8 DSpark rows; L grows left to right while the draft's calibrated survival (per position and confidence bin, greedy and sampled tables) pays for the row's marginal expert reads (one price, theta 0.04, for every batch size up to 16); dead rows reuse the anchor row's experts, so the MoE reads fewer distinct experts | `VERIFY_CAP=policy:.../r2-theta004.json,bsmax=16` | vs the round-1 batch-size threshold rule, same day, same box: greedy decode prose c4 / c8 / c16 +5 / +13 / +5 %, code +2 / +6 / +2 %, thinking c4 / c8 +4 / +2 %; c1 flat (+14 % at c1 vs no cap, round 1). Above 16 requests all 8 rows are verified (thinking c24 lost 4-7 % with the policy on) | target distribution unchanged (the width uses only drafts before the cut; Monte Carlo law test) | ours (idea from our DeepSeek-V4.1 verify cap; calibrated per-row cost rule fitted on 36K logged request-steps, after our GLM-5.3-Flash pay-for-rows rule) |
| Block verification of sampled requests under the verify cap: the accepted prefix is drawn by block verification instead of the token-by-token chain; dead drafts beyond the verify width enter the kernel as a stopped draft (zero draft probability), so dead levels are never scanned | `VCAP_BV=1` (`--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV=mask`) | thinking c1 / c8 / c24 +3.0 / +4.8 / +4.0 % (two runs vs two controls); greedy requests unchanged | target distribution unchanged (Sun et al.'s theorem over the stopped draft; joint law of the first tokens Monte Carlo tested with garbage dead rows) | **Sun et al.**, "Block Verification Accelerates Speculative Decoding" (ICLR 2025) for the algorithm; SGLang's block-verification kernel; integration under the cap ours |
| Draft statistics in one read: the per-position max draft probability that the verify rule reads and the draft probabilities q of the sampled accept, from one or two passes over the BF16 draft logits (the image's split softmax kernels) instead of FP32 copies and full-width temporaries | `VCAP_FASTPROBS=1` | thinking c8 +3.5 %, greedy decode prose c8 / c16 +1.7 / +2.6 %, code c16 +2 % | same function, FP32 summation order only; the max probability only sets the verify width | ours |
| Fused sampled accept: p and q of the drafted tokens from a per-row log-sum-exp, the accept walk (chain or block verification) and the bonus sample without materialising the [rows, vocabulary] probability tensors; from 3 requests up, self-checked against the stock accept at start-up | `VCAP_FUSED=1` | with FASTPROBS: code c16 266-270 vs 256-262 tok/s (+2 %), thinking c8 +1 to +4 % | same law; accept probabilities equal to FP32 rounding | ours |
| Draft logits through the bandwidth-efficient GEMV, FP8 copy of the draft's Markov fold matrix and of the LM head for the draft only, MXFP8 copies of the 3 draft layers | `DRAFT_BYTES=gemv,fold8,head8,draftmx8` | 40.7 -> 42.4, step -2.8 ms, acceptance unchanged (2.45-2.49 tokens/step) | draft proposals only | ours; FP8 draft head after our DeepSeek-V4.1 `DSV41_DRAFT_HEAD_FP4` |
| NVFP4 expert GEMM tiles for GB10 at decode (`tile_k` 256 instead of the RTX 5090 table's 128) | `KERNELS=moe_tiles` | MoE GEMM +5-7 % (microbench), step -0.2 to -0.5 ms | bit-identical (bench + end-to-end text) | ours; FlashInfer / CUTLASS kernels |
| Stacked Q/K/V projection of the looped layers on one Triton GEMV instead of cuBLAS at <= 16 rows per source | `KERNELS=qkvr_gemv` | 187 -> 173 us per call, step -0.4 ms | bit-identical in the GPU check | ours; SGLang's sm12x GEMV kernel |
| pico's exactly coded 12-bit BF16 GEMV (`bf16x`) for qkvr / wo and the target LM head at <= 16 rows | `PICO12=dense,lmhead` | qkvr 130 vs 174 us, wo 52 vs 66 us, LM head 2.90 vs 3.96 ms per call (M=8, cold); with the stack, step 45.8 -> 43.1 ms, speedbench quick 51.9 -> 55.0 | weights decode to the same BF16 bits; FP32 summation order differs from SGLang's GEMV | **Andy Shu (@blaesus)**, [dsgb10](https://github.com/blaesus/dsgb10) / [pico](https://github.com/Picom-code/pico-dgx-ds4) (MIT), vendored unmodified; integration ours |
| Prefill: stacked qkvr projection computed per source column block (no wasted 3x rows) | `PREFILL=qkvr_split` | that projection -63 % at 8K chunks | bit-identical at >= 128 rows | ours (found by our prefill roofline) |
| Hot-vocabulary draft: the draft proposes only from the 65,536 most frequent target tokens (ranked on 1.95M tokens of the model's own outputs for 46K prompts from open datasets, unseen ids filled by prompt-text counts), so its LM head and Markov fold read 1/3 of the rows; FP8 draft-head GEMM reads its weight once for > 64 rows (`wide8`) | `DRAFT_BYTES=...,hotvocab:65536,wide8`, `HOTVOCAB_FILE` | held-out coverage 98.8 % (speedbench 99.6 %); c1 step 39.8 -> 37.5 ms with tokens per step unchanged; speedbench quick +6 %, thinking c1 / c8 / c24 +2-9 / +4 / +5 % | draft proposals only; greedy texts byte-identical in the A/B | FR-Spec (Zhao et al., 2025) and SGLang's EAGLE token map for the idea; list, corpus and kernels ours |
| Drafter global-attention window: the draft's global layer reads a 128-token sink + the last 16,384 tokens (identity below 16,640 tokens) | `DRAFT_BYTES=...,dwin:16384` (off by default) | decode at 32K / 64K / 128K / 250K -11 / -7 / -3 / +7 %; option for workloads mostly above 200K tokens | draft proposals only; fewer far-context copies at 32K-128K | ours; attention sink from StreamingLLM (Xiao et al., 2023) |
| Routed-expert GEMM tile tables measured with real routing: NVFP4 `swap_ab=1, tile_n=32, tile_k=256` at 9-128 tokens per call, MXFP8 CuTe tactic 32x128 at 512-1536 routed rows | `MOE2=tiles` | MoE time per step c4 / c8 / c16 -3 / -4 / -6 % (microbench); greedy decode c2-c16 +2 to +6 %, thinking c8 / c24 +1 / +2 % | bit-identical (every config's outputs equal the shipped choice on real weights and routing) | ours; FlashInfer / CUTLASS kernels and tactics |
| NVFP4 expert scale codes: each expert row's E4M3 block scales stored as a row base + 5-bit codes (rare rows with > 31 distinct scales use an escape to the resident stock scales); FlashInfer's SM120 grouped-GEMM mainloop expands them into the same shared-memory scale bytes | `SCODES=1` | -4.06 % of NVFP4 expert bytes; GEMM -3 to -4.5 % per call; serving c2-c24 +1 to +3 %, thinking c24 +4 % (A/B) | bit-identical (350 / 350 tile comparisons on real weights and routing; first eager calls self-checked against stock) | ours (format idea: Christopher Owen / B12X MXFP4-CSF); FlashInfer and CUTLASS kernels (mainloop modified, see NOTICE) |
| pico stacked q/k/v without the second pass over the k/v rows: one call over the coded matrix, each column range for its own source | `PICO12=...,stack1` | -0.98 ms per c1 step (204 MB less read), 175 MB less memory | bit-identical (pico's kernel body, column ranges) | ours; pico (Andy Shu) kernel unchanged |
| Draft MXFP8 layers with narrow-N tiles (BN 16) | `DRAFT_BYTES=...,w8tile` | draft layers 155 -> 198 GB/s, -0.36 ms per c1 step | bit-identical | ours |
| Draft Markov walk: precompute the FP32 contribution of 90 % of `rnn_proj` input columns once per walk, leaving a 768x512 projection per step | `DRAFT_BYTES=...,rnnhoist` | 7-step walk, cold L2, 1 / 8 / 16 rows: 584 -> 320 / 661 -> 454 / 782 -> 596 us | draft proposals only; 99.9 % of elements correctly rounded vs FP64 | ours; walk-fusion idea from [vLLM PR #62958](https://github.com/vllm-project/vllm/pull/62958) (MIT) |
| Certified greedy LM head: a 6-bit copy of the head and a rigorous per-logit error bound select the 16-row vocabulary tiles that can hold the argmax; pico's exact GEMV computes only those; everything else is -inf before the stock softcap and argmax | `CERT_HEAD=1` | head 2.90 -> 1.67 ms per 8-row step; speedbench quick 55.0 -> 56.7 (+3 %), decode c1 +3.5 %; only greedy steps of <= 16 rows without logprobs / grammar (sampled steps run the full head) | identical argmax by construction (serving check: 45,584 rows, 0 mismatches) | ours (ported from our GLM-5.3-Flash certified head; exact pass = pico's kernel) |
| Certified greedy LM head for 17-192-row verify batches: one 6-bit screen pass over all rows, the exact packed wide-head arithmetic only on the union of candidate tiles, reduced straight to token ids | `CERT_WIDE=1` (`INKLING_CERT_WIDE`) | head + softcap + argmax at 64 rows 3.99 -> 1.72 ms, 128 rows 5.09 -> 3.19 ms (cold); serving greedy decode c4 / c8 +1.6-2.1 / +1.0-1.5 %, c1 and c16 within noise; every eligible step certified on real traffic | identical argmax to the shipped wide head (per-row proof; the GB10 tensor-core error model is qualified inside the GPU check; dense fallback when unsafe) | ours (extends the <= 16-row certified head) |
| Long-prefix fused prefill: on global layers, requests whose prefix is >= 131,072 keys run the fused prefill kernel's existing 8-warp variant (128 query rows per block), the rest the 4-warp variant, in two graph-safe launches | `INKLING_ATTN_W8_FROM` (default 131072; 0 = one launch) | the 4-warp kernel loses L2 reuse of K/V on long sweeps (70.7 TF/s at 120-256K, 46.5 at 1M); per global layer and 8K chunk: 512K 1,082 -> 777 ms (1.39x), 1M 2,281 -> 1,547 ms (1.47x) | bit-identical to the single 4-warp launch (all rows, output and LSE, 256K-1M, mixed batches, graph replay) | ours |
| Packed 12-bit GEMM for 17-256-row verify batches: reads pico's coded weights once per call, mma.sync over all rows (qkvr <= 96 rows per source, wo_ud <= 32, LM head <= 256; verify/decode forwards only) | `WIDE=dense,lmhead` | qkvr 1.24-1.34x and LM head 1.34-1.67x faster than cuBLAS at 17-64 rows; thinking c8 +8 %, greedy decode c4 / c8 / c16 +4 / +5 / +3 % | weights decode to the same BF16 bits; FP32 summation order differs from cuBLAS (error vs FP64 equal or lower) | ours; pico's `bf16x` reader (Andy Shu, MIT) |
| Verify attention on the split-KV relative-bias kernel at every batch size (cap 32 -> 384 rows) and the bias computed in-kernel above 32 rows, instead of padded FA4 with a sheared [t, 32, 1280] bias | `VATTN=1` | verify-attention visit 1.4-1.7x faster at 8-48 requests; thinking c24 +7 %, greedy decode c8 / c16 +5 / +6 % | same math; bias bit-identical, output within 1 BF16 ulp of the stock compact path | ours; SGLang's Inkling2 verify kernel |
| Packed 12-bit global-layer KV: full, committed KV pages of the 8 global-layer visits are also kept as sign + 7-bit mantissa bytes and a 4-bit exponent code against a fixed 15-exponent table (escape stream for the ~0.03 % of values outside it), 193 instead of 256 bytes per key and head; decode / verify attention decodes packed tiles in registers; 2,048 side slots per layer follow the pages being decoded (LRU) | `KV12=1`, `KV12_SLOTS=2048` | global verify visit 1.15-1.28x at 128K-256K (microbench); c1 step at 128K 62.2 -> 56.8 ms, at 250K 80.1 -> 75.2 ms; 92.6-99.9 % of tiles read packed; +6.1 GiB | bit-identical: decoded tiles equal the BF16 cache, attention partials equal stock's | ours, on SGLang's split-KV decode kernel; inspired by lossless exponent coding of floats (ZipNN, SplitZip) and by pico's exactly coded 12-bit `bf16x` weight format (Andy Shu); no code taken from either |
| Prefill attention with the relative-position bias computed inside the kernel (FA2-style, mma.sync, paged KV, sliding window and global layers), instead of a materialised [t, 24, 1024] bias read by FA4 | `ATTN=prefill_swa,prefill_global` | SWA attention 3.3x vs the round-1 bias_layout path; cold prefill 8K / 32K / 128K +15 / +18 / +26 % | same math; error vs FP32 reference equal to stock's | ours (FlashAttention-2 algorithm, Tri Dao) |
| Prefill glue fusion: the loop-message mixes, attention-residual mix + RMSNorm and the MLP short conv + residual update in fused kernels that skip the FP32 [T, 2304] round trips | `PFUSE=all` | -187 ms per 8K chunk (microbench); with the fused attention: cold prefill 1K / 8K / 32K / 128K +14 / +29 / +30 / +36 % | same per-element math; only FP32 row-sum order differs (error vs FP64 equal or lower), short conv bit-identical | ours |
| Prefill: relative-position bias written directly in FA4's sheared layout | `PREFILL=bias_layout` | with quant_once: 8K prefill 3,781 -> 4,253 tok/s | bit-identical (attention output and LSE) | ours |
| Prefill: MoE input quantised to NVFP4 once per token, then gathered to its 8 experts | `PREFILL=quant_once` | (with bias_layout, above) | bit-identical | ours |
| Prefill: MoE combine writes BF16 directly | `PREFILL=partial_bf16` | combine -22 to -32 %; with qkvr_split: 8K 3,408 -> 3,781 tok/s | bit-identical | ours |
| SGLang DSpark cookbook cell (FA4, page 128, 8 draft tokens, 4 `SGLANG_OPT_*` switches) | always | base | - | SGLang |
| One-pass checkpoint load: SGLang reads the safetensors file directly and large frees return to the OS before KV allocation | `LOAD_NOMMAP=1` | 121-137 s warm start-up in the full benchmarks | weights unchanged | SGLang's loader flag; [sxuff](https://github.com/sxuff/inkling-2-mini-spark) and [0xSero](https://github.com/sybil-solutions/inference-optimisation-skill) for the load-strategy idea |
| KV pool sizing | `MEM_FRACTION=0.93`, `SWA_RATIO=0.05` | 1.55M global-layer KV tokens with 5.8-6.1 GiB MemAvailable after start | exact | ours |

Tried and not adopted (details in [docs/experiments.md](docs/experiments.md)): vLLM (same decode, slower prefill,
half the free memory), `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` (flat), fixed verify caps (lose on concurrent
thinking), the Triton port of a packed-BF16 GEMV (superseded by pico's kernel), NVFP4 prefill tile retunes (<= 2 %),
other draft block sizes (refused by the draft), L2 prefetch / side-stream L2 warming before the small per-visit kernels
(exact, but GB10 drops `prefetch.L2` and a warm read only moves the same DRAM bytes: -0.1 ms per step), a native
layer-stack backend under SGLang (best case 1.13x at c4-c24 after the overlays, below our 1.20x bar), the verify
policy above 16 requests (thinking c24 -4 to -7 %), KV page size 16 (flat decode; the fused prefill attention needs
page 128), a stream-K packed GEMM for 97-192-row calls (1.2-1.45x cuBLAS in isolation, flat to negative in serving),
further prefill glue fusion (-70 ms per 8K chunk in isolation, flat end to end), a draft sampling temperature below
the request's (no gain or worse on thinking), a 4-bit draft MLP (same step time; its tokens-per-step gain is within
boot-to-boot variation and the offline acceptance change is -0.17 % with a confidence interval that includes 0),
late kill of doomed verify rows from a mid-stack readout (the readouts barely separate doomed rows: AUC 0.59-0.74),
shared-prefix KV reads for concurrent requests (L2 already absorbs the reuse), a fused routed-expert epilogue (already
fused on GB10), a streaming NVFP4 MoE kernel (bit-identical and 2-5 % faster than stock per call, not faster than the
scale codes), a centroid-clustered draft vocabulary (at most ~0.2 ms per c1 step in the byte model), verify-policy
retunes (theta 0.03 / 0.06, a second price above 16 requests: -1.4 % to flat), an 80K-token hot vocabulary.

## The model

Read from the checkpoint's `config.json` at revision `ed40e7da`:

| | |
|---|---|
| Parameters | 26.4B total, 3.4B active per pass (model card) |
| Layers | 24 stored; layers 6-17 form a looped core that runs 3 times, so a token passes 48 layer applications |
| Experts | 128 routed experts per MoE layer, top-8, no shared expert |
| Attention | 20 sliding-window layers (window 1,024) and 4 global layers (5, 11, 17, 23); 24 query heads, 8 KV heads, head dim 128, relative position bias; a short convolution (kernel 4) in every layer. Each pass of the looped core keeps its own KV and convolution state |
| Context | up to 1,048,576 tokens |
| Precision | routed experts NVFP4 in layers 0-19 and MXFP8 in layers 20-23 (`quant_method: inkling2`, ModelOpt); attention, embeddings and LM head BF16 |
| Draft model | DSpark, 3 layers, block size 8 (`mtp.safetensors`, 0.64 GB) |
| Inputs | text and images (23-layer vision tower); text out. This recipe serves text only |
| Reasoning | always thinks; `reasoning_effort` none / minimal / low / medium / high / xhigh / max or 0.0-0.99 (default high = 0.9) |
| On disk | 20.2 GB |
| Licence | Apache-2.0 |

## Running it

Requirements: one DGX Spark with DGX OS, Docker with the NVIDIA runtime (the default on DGX OS), ~80 GB of disk for
the image (56 GB) and the weights (20 GB), and nothing else large running on the box.

```bash
git clone https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark && cd Inkling-2-Mini-1x-DGX-Spark
scripts/download.sh                       # weights at the pinned revision -> ~/models/Inkling-2-Mini, sha256-checked
docker pull lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de
scripts/launch.sh up                      # SGLang + DSpark + overlays on 127.0.0.1:8000, mem 0.93, SWA ratio 0.05
#   or: OVERLAY=0 LOAD_NOMMAP=0 MEM_FRACTION=0.62 SWA_RATIO=0.1 scripts/launch.sh up  # stock benchmark cell
scripts/launch.sh status                  # container, API, MemAvailable
python3 bench/sanity.py all               # chat, effort ladder, tool call, JSON schema, 120K needle
scripts/launch.sh down
```

The Chat Completions API is at `http://127.0.0.1:8000/v1`, model name `inkling-2-mini`. The server listens on
loopback only; from another machine use `ssh -L 8000:127.0.0.1:8000 <spark>`, or set `HOST=0.0.0.0` on a trusted LAN.
Do not forward the port through your router.

The first start builds the overlay CUDA extensions (pico12, wide, certified head, prefill attention; ~30 s each) and
compiles the overlay Triton kernels; all are cached in `~/.cache/inkling-2-mini-sglang` like SGLang's own JIT caches.

Do not start the server while a large download or copy is still running: on GB10 the page cache and the GPU share
the same memory (the "page-cache trap", see Credits). `scripts/launch.sh up` refuses to start next to another model
server and runs `scripts/memguard.sh` for the first 20 minutes. The guard stops the container if MemAvailable falls
below 3.5 GiB for two consecutive samples, 2 s apart. An out-of-memory on GB10 can hang the box until a power cycle.

### Using it

Reasoning effort goes into the chat template, either as the top-level `reasoning_effort` field or as
`chat_template_kwargs: {"reasoning_effort": ...}`; both reach the template. Values: `none` (0.0), `minimal` (0.1),
`low` (0.2), `medium` (0.7), `high` (0.9, the default), `xhigh` / `max` (0.99), or a number in [0.0, 0.99]. The model
always writes a reasoning block, even at `none`; the reasoning parser returns it as `reasoning_content`, the answer as
`content`. Tools use `tools` / `tool_calls`. Structured output uses `response_format` with a JSON schema (xgrammar).

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
  "reasoning_effort": "low", "max_tokens": 1024}'
```

The checkpoint's `generation_config.json` sets temperature 1.0; pass `temperature` per request if you want
something else. Reasoning tokens count toward `max_tokens`: at effort high, leave several thousand.

### Every flag explained

`scripts/launch.sh up` runs one container (the command it prints, with the defaults):

```bash
docker run -d --name inkling-2-mini --gpus all --ipc=host --network host --shm-size 16g --restart no \
  -v ~/models/Inkling-2-Mini:/model:ro -v ~/.cache/inkling-2-mini-sglang:/root/.cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v $PWD/overlay:/opt/inkling-overlay:ro -e PYTHONPATH=/opt/inkling-overlay \
  -e INKLING_VERIFY_CAP=policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json,bsmax=16 \
  -e INKLING_VERIFY_CAP_LOG_EVERY=0 \
  -e INKLING_DRAFT_BYTES=gemv,fold8,head8,draftmx8,hotvocab:65536,wide8,w8tile,rnnhoist \
  -e INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json \
  -e INKLING_KERNELS=moe_tiles,qkvr_gemv \
  -e INKLING_MOE_TILES='1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256' \
  -e INKLING_PICO12=dense,lmhead,stack1 \
  -e INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16 -e INKLING_PREFILL_QKVR_OUT=strided \
  -e INKLING_CERT_HEAD=1 -e INKLING_WIDE=dense,lmhead -e INKLING_WIDE_SCOPE=verify -e INKLING_VATTN=1 \
  -e INKLING_ATTN=prefill_swa,prefill_global -e INKLING_PFUSE=all -e INKLING_MOE2=tiles -e INKLING_SCODES=1 \
  -e INKLING_VERIFY_CAP_FASTPROBS=1 -e INKLING_VERIFY_CAP_FUSED_ACCEPT=1 -e INKLING_KV12=1 -e INKLING_KV12_MAX_PAGES=2048 \
  -e INKLING_CERT_WIDE=1 -e INKLING_KV12S=0 -e INKLING_KV12S_OVERFLOW_HEADS=512 -e INKLING_KV12S_DRAFT_OVERFLOW_HEADS=2048 \
  -e MAX_JOBS=4 \
  -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \
  -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \
  -e MALLOC_MMAP_THRESHOLD_=1048576 \
  --entrypoint python3 lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de \
  -m sglang.launch_server --model-path /model --served-model-name inkling-2-mini \
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 --speculative-use-block-verification \
  --mem-fraction-static 0.93 --context-length 262144 \
  --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2 \
  --enable-metrics --host 127.0.0.1 --port 8000 --weight-loader-disable-mmap --swa-full-tokens-ratio 0.05 \
  --max-mamba-cache-size 512 --max-prefill-tokens 8192
```

Engine flags (the SGLang cookbook's DGX Spark DSpark cell, text only):

| flag / variable | why |
|---|---|
| `lmsysorg/sglang@sha256:efac853b…` | the `inkling-2-mini` tag pinned by digest (multi-arch; the Spark pulls arm64 `sha256:bc811ac7…`); SGLang `0.0.0.dev1+g03fde9cc3` |
| `--gpus all --ipc=host --shm-size 16g` | the GB10, and shared memory for the scheduler / detokenizer processes |
| `--network host` + `--host 127.0.0.1` | the API listens on loopback only |
| `--restart no` | nothing restarts behind your back after an out-of-memory |
| `-v …:/model:ro`, `HF_HUB_OFFLINE=1` | local weights at the pinned revision; no Hugging Face access at start |
| `-v ~/.cache/inkling-2-mini-sglang:/root/.cache` | keeps the JIT / Triton / FlashInfer caches and the overlay extension builds for warm starts |
| `MAX_JOBS=4` | caps parallel JIT compile jobs during the first start (cookbook) |
| `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1` | DSpark verify attention through the split-KV relative-bias decode kernel (cookbook) |
| `SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1`, `…_PREPARE_INPUTS_ONCE=1`, `…_SKIP_PREFILLED_INPUTS=1` | DSpark step glue inside the CUDA graphs (cookbook) |
| `--tp 1 --ep 1` | one GPU |
| `--attention-backend fa4` | FlashAttention-4 path with Inkling's relative bias (the only backend Inkling2 accepts) |
| `--dtype bfloat16` | activations; the weights stay NVFP4 / MXFP8 as shipped |
| `--page-size 128` | KV page size (cookbook) |
| `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8` | the bundled 3-layer DSpark draft; the block size is fixed at 8 by the trained head |
| `--speculative-use-block-verification` | block verification for sampled requests (set by `VCAP_BV=1` with the overlays on; not part of the cookbook cell) |
| `MEM_FRACTION=0.93` (`--mem-fraction-static 0.93`) | static pool for weights, KV and CUDA graphs. Gives 1.55M global-layer KV tokens (see Memory budget) |
| `SWA_RATIO=0.05` (`--swa-full-tokens-ratio 0.05`) | sliding-window pool as a fraction of the global-layer pool. The image defaults to 0.1 |
| `MIN_AVAIL_GIB=3.5` | host MemAvailable floor for the memory guard. Accepts decimal GiB |
| `--max-mamba-cache-size 512`, `--max-prefill-tokens 8192` | conv-state slots (the image's default sized 1,232; peak use measured 76) and one 8,192-token prefill chunk in flight instead of two: +2.2 % KV tokens, prefill and decode unchanged |
| `LOAD_NOMMAP=1` (`--weight-loader-disable-mmap`, `MALLOC_MMAP_THRESHOLD_=1048576`) | one-pass safetensors read; large frees return to the OS before SGLang sizes the KV pool; `0` uses the stock mmap loader |
| `--context-length 262144` | per-request limit; the KV pool is shared, so this reserves nothing per request. The model supports 1,048,576 |
| `--grammar-backend xgrammar` | JSON-schema / regex constrained output |
| `--reasoning-parser inkling`, `--tool-call-parser inkling2` | thinking to `reasoning_content`, answer to `content`, tool calls to `tool_calls` |
| `--enable-metrics` | Prometheus `/metrics` (speedbench reads them) |

Overlay switches (`OVERLAY=0` turns all of them off; each is described in `overlay/<package>/README.md`):

| launch.sh variable (container env) | default | what |
|---|---|---|
| `OVERLAY_DIR` (mounted at `/opt/inkling-overlay`, on `PYTHONPATH`) | `overlay/` of this checkout | the overlay packages and the loader |
| `VERIFY_CAP` (`INKLING_VERIFY_CAP`) | `policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json,bsmax=16` | calibrated verify-width rule (policy file: per-position survival tables and the price theta); `bsmax=16`: all 8 rows above 16 requests; the round-1 `cost:1=0.30,2=0.25,4=0.15,8=0.0` still works; `off` = stock |
| `VCAP_BV` (`--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV`) | `1` (`mask`) | block verification for sampled requests under the verify cap; `mask` hands the kernel the dead drafts as a stopped draft, `stock` runs the kernel over all rows and caps after (both exact); `0` = chain accept |
| `VCAP_FASTPROBS` (`INKLING_VERIFY_CAP_FASTPROBS`) | `1` | max draft probabilities for the verify rule and the sampled accept's draft probabilities from the BF16 draft logits in one or two reads |
| `VCAP_FUSED` (`INKLING_VERIFY_CAP_FUSED_ACCEPT`) | `1` | sampled accept (chain or block verification) computed from the logits for decode batches of >= 3 requests (`INKLING_VERIFY_CAP_FUSED_ACCEPT_MIN_BS`); self-checked at start-up, stock accept otherwise |
| `DRAFT_BYTES` (`INKLING_DRAFT_BYTES`), `HOTVOCAB_FILE` | `gemv,fold8,head8,draftmx8,hotvocab:65536,wide8,w8tile,rnnhoist`, `.../inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json` | draft logits on the GEMV, FP8 Markov fold, FP8 draft LM head, MXFP8 draft layers, draft vocabulary = the first 65,536 ids of the ranked list, single-read FP8 draft GEMM above 64 rows, narrow-N tiles for the MXFP8 draft layers, precomputed Markov projection partials; `dwin:W[:S]` is optional (off by default) |
| `KERNELS` (`INKLING_KERNELS`), `MOE_TILES` | `moe_tiles,qkvr_gemv` | GB10 NVFP4 tile table for decode; stacked Q/K/V on one GEMV |
| `PICO12` (`INKLING_PICO12`) | `dense,lmhead,stack1` | pico's 12-bit packed GEMV for qkvr / wo and the target LM head at <= 16 rows; `stack1`: the stacked q/k/v of the looped layers in one pass over the coded matrix (+1.7 GB memory for the packed copies; the extensions build in ~30-35 s each on the first start) |
| `PREFILL` (`INKLING_PREFILL`) | `qkvr_split,bias_layout,quant_once,partial_bf16` | prefill: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once per token, BF16 MoE combine |
| `CERT_HEAD` (`INKLING_CERT_HEAD`) | `1` | certified greedy LM-head argmax (6-bit screen + pico's exact GEMV on the candidate tiles; +358 MB); `check` serves the full head and counts mismatches; `0` = off. Needs `PICO12` with `lmhead` |
| `WIDE` (`INKLING_WIDE`), `WIDE_SCOPE` | `dense,lmhead`, `verify` | packed 12-bit GEMM for 17-256-row calls (verify batches of 3-32 requests) on decode / verify forwards; per-kind row caps `INKLING_WIDE_MAX_M_KIND` (default `qkvr=96,wo=32,head=256`) |
| `VATTN` (`INKLING_VATTN`) | `1` | verify attention on the split-KV relative-bias kernel for up to 48 requests, bias in-kernel above 32 rows (`cap` = only the row cap) |
| `KV12` (`INKLING_KV12`), `KV12_SLOTS` (`INKLING_KV12_MAX_PAGES`) | `1`, `2048` | packed 12-bit copy of full, committed global-layer KV pages, read by decode / verify attention; 2,048 slots per global layer = 6.1 GiB, enough for one 256K-token request (`0` = one slot per page of the pool) |
| `ATTN` (`INKLING_ATTN`) | `prefill_swa,prefill_global` | fused relative-bias attention for prefill / extend (CUDA extension, builds in ~30 s on the first start) |
| `SCODES` (`INKLING_SCODES`) | `1` | NVFP4 expert scale codes (+0.81 GB: the stock scales stay resident for escapes and fallback; first start builds the modified FlashInfer kernels, ~4.5 min, then cached) |
| `MOE2` (`INKLING_MOE2`) | `tiles` | tile tables for the routed-expert GEMMs at verify batch sizes (over `moe_tiles`); `INKLING_MOE2_NVFP4` / `INKLING_MOE2_MXFP8` override the tables |
| `PFUSE` (`INKLING_PFUSE`) | `all` | prefill glue fusion (`loopmix,resnorm,sconvres`), extend batches of >= 512 rows only |
| `CERT_WIDE` (`INKLING_CERT_WIDE`) | `1` | certified greedy head for 17-192-row verify batches (needs `CERT_HEAD`, `PICO12` and `WIDE` with `lmhead`) |
| `INKLING_ATTN_W8_FROM` | `131072` | prefix length (keys) from which global-layer prefill uses the 8-warp fused kernel; `0` = one 4-warp launch |
| `KV12S` (`INKLING_KV12S`) | `0` | opt-in packed global KV store (see "Packed global KV store"); `1` turns `KV12` off, adds `--enable-unified-memory --cuda-graph-backend-prefill=disabled --max-running-requests 48 --cuda-graph-backend-decode=full --cuda-graph-max-bs-decode=48` and uses `MEM_FRACTION=0.91` unless set |

Left at SGLang's defaults (measured, see docs/experiments.md): chunked prefill 8,192 tokens (16,384: +2-3 % at
32K-128K, more memory), up to 48 running requests (CUDA graphs captured to 48), radix prefix cache on, overlap
scheduler on, prefill CUDA graphs on.

### Memory budget

GB10 has one 121.6 GiB pool for the GPU and the operating system. `MEM_FRACTION=0.93` budgets for the weights,
KV and CUDA graphs. With `LOAD_NOMMAP=1`, large frees from checkpoint loading return to the OS before SGLang
sizes the static pool. The packed KV overlay then allocates its side pool.

| | |
|---|---|
| KV cache, global layers (and the draft's global layer) | 1,593,088 tokens (Spark_05, with the two memory flags above; 1,551,104 / 1,557,120 without), BF16 |
| packed 12-bit KV side pool (global layers) | 8 x 2,048 slots of 128 tokens, 6.1 GiB: covers the pages being decoded, up to one 256K-token request |
| KV cache, sliding-window layers | ~77.5K tokens at `SWA_RATIO=0.05`. With radix caching on, 48 requests at one 1,024-token window plus a 128-token page each and two 8,192-token prefill chunks need ~72K tokens |
| MemAvailable after start / lowest under stress | 6.3 GiB / 5.2 GiB (Spark_05; v2.6 build without the flags 5.8-6.1 / 5.2-5.4) |

That capacity is equivalent to 5.9 requests at the full 262,144-token context, or ~47 agents with 32K each.
The sweep targeted at least 3.5 GiB MemAvailable under stress, a level that held up in memory-cliff measurements
on other GB10 recipes. MemAvailable after start varied by up to ~1 GiB between boots. At 0.94, Spark_06 came
within 0.2 GiB of the floor. The 0.93 default leaves more room for that variation. [Full memory sweep](docs/experiments.md#v26-kv-pool-sizing)

To leave more RAM for other work, lower `MEM_FRACTION`. Each 0.01 frees ~1.06 GiB at a cost of ~20-25K
global-layer KV tokens.

### What differs from the published recipes, and why

| | SGLang cookbook (DGX Spark, DSpark) | vLLM recipe (DGX Spark) | here |
|---|---|---|---|
| engine | SGLang | vLLM | **SGLang**: the stock engines decode at the same speed on one Spark (c1 within 3 %, c16 / c24 within run noise); SGLang prefills 1.3-3.4x faster, keeps long prompts in its prefix cache more reliably and leaves twice the host memory ([docs/experiments.md](docs/experiments.md)) |
| memory | `--mem-fraction-static 0.45` (518K KV tokens) | `--gpu-memory-utilization` unset (vLLM's own suggestion would leave the host 0-2 GiB) | `MEM_FRACTION=0.93` and `SWA_RATIO=0.05` give 1.55M global-layer KV tokens with a 6.1 GiB packed side pool, 5.2-5.4 GiB MemAvailable under stress and a 3.5 GiB guard floor |
| context | not set (model default) | 262,144 | 262,144 (the model supports 1M; see "Longer context") |
| multimodal | `--enable-multimodal` | on | off (text only; add `--enable-multimodal` for images) |
| images, weights | pulled by tag, weights from the Hub at start | same | pinned image digest, weights at a pinned revision, sha256-checked, offline at start |
| bind address | 0.0.0.0 | 0.0.0.0 | 127.0.0.1 |
| speed overlays | - | - | thirteen packages, on by default (`overlay/`); `OVERLAY=0` disables them |

### Longer context: 512K and 1M

The model supports 1,048,576 tokens; `CTX` sets the per-request limit (default 262,144). The KV pool is shared, not
reserved per request. Measured with the default store (`KV12S=0`, v2.6 build on Spark_05; the v2.7 long-prefix fix makes
the cold prefills below faster):

| `CTX` | needle | cold prefill | decode at that context | lowest MemAvailable |
|---|---|---:|---:|---:|
| 524288 | PASS at 499,995 tokens (also with thinking c24 alongside) | 499,995 tokens in 332-351 s | 24.5 tok/s at 491K (112 ms step) | 5.5 GiB |
| 1048576 | PASS at 999,296 tokens (thinking c24 alongside) | 1,024,204 tokens in 1,285 s | 14.0 tok/s at 1.02M (200 ms step) | 4.9 GiB |

A single 1M prompt takes two thirds of the default pool and, while it prefills, concurrent requests slow down (thinking
c24 fell to 62 tok/s aggregate). With the packed store (`KV12S=1`) the 1M option is not verified yet.

### Packed global KV store (opt-in, `KV12S=1`)

`KV12S=1` stores the global-layer KV only in the exactly coded 12-bit form that `KV12` reads (sign and mantissa
bytes, a 4-bit exponent code with 64 escape entries per page and head, a raw slot for the rare overflowing head) and
drops both the BF16 global pool and the 6.1 GiB side copy. The open page of each request stays BF16 until it is full
and committed; the draft's global layer is kept raw BF16 inside the same page envelope.

| | default (`KV12S=0`) | `KV12S=1` |
|---|---:|---:|
| global-layer KV tokens | 1.59M | **2,168,320** |
| MemAvailable after start / lowest in the run | 6.3 / 5.2 GiB | 6.9 / 4.2 GiB |
| speedbench quick c1 | 63.49 | 63.16 (same tokens per step) |
| sanity, needles 120K / 240K | 11/11, PASS | 11/11, PASS |

What is verified: every store and read of the packed pages is bit-identical to SGLang's BF16 path in the GPU checks
(decode and verify partials incl. CUDA-graph replay, speculative accept / reject across pages, staging and overflow,
both prefill loaders), and the packed decode reads are 1.16-1.23x faster than the BF16 ones at 32K-128K. In serving,
15.5M page-heads were packed with no escape overflow on the target layers.

Known issues, to be fixed in v2.7.1, which is why it is not the default:

- Exactness of the served text is under investigation: greedy speedbench quick texts match v2.6 on 27 of 32
  prompts (the default store matches 32/32). The difference is deterministic and not explained yet; the target
  model's weights and arithmetic are unchanged.
- Cold prefill above ~120K tokens was about 2x slower in the first build (120K needle 72 s vs 33 s); the shipped
  build reads the packed prefix through a bounded BF16 scratch, which is bit-identical but not yet measured end to end.
- The 512K / 1M option is not verified with it yet.

### FP8 KV

More concurrent long requests can also use an FP8 KV cache. FP8 KV changes the model's numerics (lossy) and is
not part of this recipe's measured profile; Tech2Wild's vLLM recipe measured an FP8 tier (3.3M tokens, needle PASS at
~950K, its tool-calling eval 87.0 vs 88.4 with BF16 KV) and documents the memory budget for it.

## Reproducing the numbers

```bash
git clone https://github.com/knapcio/speedbench speedbench && git -C speedbench checkout 6733653
MEM_FRACTION=0.94 scripts/launch.sh up     # full speed benchmark settings
#   or: OVERLAY=0 LOAD_NOMMAP=0 MEM_FRACTION=0.62 SWA_RATIO=0.1 scripts/launch.sh up  # stock benchmark cell
bench/full.sh MYRUN                       # ~40 min, writes results/MYRUN/
python3 bench/report.py results/MYRUN      # README-style tables
bench/screen.sh MYSCREEN                   # ~10 min screen: speedbench quick, thinking c8, 32K prefill
```

The overlays' own checks run inside the image with the server stopped: `overlay/tools/gpu_check_qkvr.py`,
`gpu_check_pico12.py`, `gpu_check_prefill.py`, `gpu_check.py` (draft package), `gpu_check_bv.py`,
`gpu_check_fastprobs.py`, `gpu_check_fused_accept.py` (sampled accept) and the CPU unit tests in `overlay/tests/` (see
`overlay/README.md`).

## Licences

Repository-authored scripts, overlays, tests and documentation are MIT (see `LICENSE`). `overlay/inkling_pico12/
third_party/pico/` holds the GEMV sources by Andy Shu (dsgb10 / pico), unmodified, under their MIT licence (`LICENSE` next to
them); `overlay/inkling_wide/csrc/bf16x_format.cuh` copies pico's `bf16x` reader under the same licence
(`overlay/inkling_wide/third_party/pico/LICENSE`); the packed-KV attention kernel in `overlay/inkling_kv12/ops.py`
follows SGLang's Inkling2 split-KV decode kernel (Apache-2.0) with its K / V tile loads replaced; see `NOTICE`. The serving images (SGLang, vLLM) are Apache-2.0 and are used as published, pinned by digest.
The Inkling-2-Mini weights are Apache-2.0 from Thinking Machines Lab.

See [CREDITS.md](CREDITS.md) for the work this builds on.
