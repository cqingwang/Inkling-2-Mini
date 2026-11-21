# Experiments: engines and levers

Everything that was measured to pick the profile in the README: the SGLang vs vLLM comparison, every lever that
was screened, and what was rejected and why. Raw files are in `results/screen/` (screening) and `results/<label>/`
(full runs).

## Setup

- Two DGX Spark (GB10, 121.6 GiB unified memory, driver 580.178.04, kernel 6.17.0-1032-nvidia), used standalone
  (`Spark_05`, `Spark_06`); one model server per box, nothing else running. GPU clock capped at 2200 MHz by the
  hosts' own timer (not changed for these runs).
- Model `thinkingmachines/Inkling-2-Mini` revision `ed40e7da` (sha256-checked), bundled DSpark draft.
- SGLang image `lmsysorg/sglang:inkling-2-mini` (`sha256:efac853b…`, arm64 `sha256:bc811ac7…`).
- vLLM image `vllm/vllm-openai:inkling-2-mini` (`sha256:c2dc759f…`, arm64 `sha256:ecdf0ccc…`, vLLM
  `0.1.dev25855+g7cb5dae22`).
- Benchmarks run on the box itself against 127.0.0.1.

### Screening set (`bench/screen.sh`, ~10 min per arm)

1. speedbench 0.3 quick: 30 frozen prompts x 160 tokens, c1, temperature 0, reasoning effort low. Geometric mean of
   decode tok/s over the prompts; the engine counters give tokens per step and step time.
2. Thinking load: 8 concurrent requests, 16 seeded arithmetic puzzles, 2,048 tokens each, effort high,
   temperature 0.8 (`bench/think_load.py`). Aggregate completion tok/s.
3. speedbench longctx at 32K: cold time to first token (prefill tok/s), cached TTFT, decode at 32K context
   (2 documents).

A screen is one run per arm and is only a filter. Differences under ~3 % at c1 (speedbench quick's MDE) and under
~5 % on the sampled thinking load are noise at this stage; winners were confirmed with repeated full runs on both
boxes (below).

## Screening results

One run per arm, `bench/screen.sh`. speedbench quick = geometric mean of decode tok/s over 30 prompts (c1, T=0,
effort low); thinking c8 = aggregate tok/s; prefill / decode at 32K from speedbench longctx (cold TTFT of a
32,702-token prompt; decode over 2 documents).

| arm | box | speedbench quick all (prose) | tokens / step | thinking c8 | 32K prefill tok/s | 32K decode tok/s | KV pool, tokens | MemAvailable after start | start-up |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 vLLM, recipe settings (util 0.80) | 06 | 42.82 (33.32) | 2.448 | 199.6 | 2,429 | 53.7 | 1,347,359 | 20.0 GiB | 313 s (cold) |
| V2 vLLM + `--max-num-batched-tokens 8192` | 06 | 41.47 (32.45) | 2.448 | 195.0 | 2,763 | 53.1 | 693,420 | 24.9 GiB | 360 s |
| S1 SGLang, cookbook cell, mem 0.62 | 05 | 41.86 (33.18) | 2.448 | 189.3 | 3,245 | 51.9 | 844,928 | 39.0 GiB | 378 s (cold) |
| S2 SGLang + `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1` | 05 | 41.07 (32.59) | 2.448 | 191.6 | 3,255 | 57.2 | 845,952 | 39.9 GiB | 237 s (warm) |

- vLLM's `--max-num-batched-tokens 8192` lifts 32K prefill by 14 % (vLLM caps scheduled tokens at 2048 with
  speculative decoding) but halves the reported KV pool (the per-request sliding-window and conv blocks grow with
  the scheduled tokens); decode unchanged. Not adopted (vLLM is not the default engine).
- The NV-mega bundle (glue graphs, folded accept tail, deferred draft KV, blockwise rejection) is flat: the step is
  GPU-bound (profile: GPU busy 99.4 %, host gap 0.39 ms per step).

## SGLang vs vLLM, same benchmark, both engines on both boxes

Full runs (`bench/full.sh`), stock recipe settings of each engine (SGLang: cookbook DSpark cell with mem 0.62;
vLLM: official recipe + `--gpu-memory-utilization 0.80`, `--enable-auto-tool-choice --tool-call-parser inkling`),
F1 = SGLang on Spark_05 / vLLM on Spark_06, F2 = swapped.

speedbench full (73 prompts x 256 tokens, c1, T=0, effort low), decode tok/s geomean:

| engine | box | prose | code | structured | agentic | multilingual | long context | all 73 | TTFT median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGLang | 05 | 33.0 | 48.5 | 44.9 | 49.9 | 25.6 | 51.5 | **41.09** | 182 ms |
| SGLang | 06 | 32.3 | 47.5 | 43.8 | 48.7 | 25.0 | 50.4 | **40.16** | 186 ms |
| vLLM | 06 | 32.8 | 47.9 | 42.4 | 52.7 | 25.0 | 54.4 | **41.11** | 163 ms |
| vLLM | 05 | 33.0 | 48.2 | 42.7 | 53.2 | 25.2 | 54.8 | **41.38** | 163 ms |

Decode table (T=0, 256 tokens, effort none; c1 per-stream, higher concurrency aggregate):

| | prose c1 / c4 / c16 | code c1 / c4 / c16 | json c1 / c4 / c16 |
|---|---|---|---|
| SGLang 05 | 33.6 / 76.0 / 190.8 | 44.1 / 89.0 / 221.3 | 47.0 / 90.6 / 236.3 |
| SGLang 06 | 32.8 / 74.4 / 181.5 | 43.1 / 85.3 / 221.3 | 46.0 / 88.1 / 228.0 |
| vLLM 06 | 32.6 / 75.6 / 197.1 | 38.5 / 89.7 / 231.2 | 41.1 / 93.0 / 240.8 |
| vLLM 05 | 32.8 / 73.9 / 194.2 | 38.7 / 87.4 / 214.5 | 40.5 / 93.6 / 236.6 |

Thinking load (effort high, T=0.8, aggregate tok/s): SGLang c1 53.3 / 51.4, c8 198.1 / 189.1, c24 380.9 / 368.9;
vLLM c1 54.7 / 54.7, c8 201.8 / 213.7, c24 397.1 / 381.0 (05 / 06 order as above; one run each, sampled).

Cold prefill (speedbench longctx, tok/s; TTFT in brackets):

| engine | 1K | 8K | 32K | 128K | 128K cached TTFT |
|---|---:|---:|---:|---:|---:|
| SGLang (05 / 06) | 2,190 / 2,137 (0.5 s) | 3,400 / 3,399 (2.4 s) | 3,234 / 3,230 (10.1 s) | 2,419 / 2,426 (54 s) | 27 s (1 of 2 hits) |
| vLLM (06 / 05) | 644 / 649 (1.6 s) | 1,803 / 1,762 (4.5 s) | 2,437 / 2,427 (13.4 s) | 1,979 / 1,952 (66 s) | 66 s (0 of 2 hits) |

Memory after start-up: SGLang ~39-40 GiB MemAvailable with an 845K-token KV pool; vLLM ~20 GiB with a 1.35M-token
pool (its own suggestion, `--gpu-memory-utilization` unset, would leave the host 0-2 GiB). Start-up: vLLM 5.2-5.6 min,
SGLang 6.2 min cold / 3.7-4 min warm.

**Verdict.** Decode is a tie: c1 within 3 % on both boxes (vLLM +0.7 %/+3 %), c16 and thinking c24 within run
noise, vLLM slightly ahead at high concurrency in some cells and SGLang ahead on code/json c1. SGLang prefills
1.3x (32K) to 3.4x (1K) faster, serves cached long prompts more reliably, and leaves twice the host memory. SGLang is
the base of this recipe; all further work is on SGLang. This agrees with thefullctx's head-to-head (vLLM +5 % on
short c1 decode, SGLang ~2x long prefill, much less memory).

## Where the decode time goes (c1)

Torch profile of the live engine (SGLang `/start_profile`, 40 steps, greedy prose, DSpark on; analysis script and
tables in `docs/profile/`): a c1 step is 60.4 ms = draft 9.7 + target verify 50.5 ms; the GPU is busy 99.4 % of the
wall time (host gap 0.39 ms per step, no graph breaks). Per step: routed MoE 30.2 ms (NVFP4 26.4 + MXFP8 3.8; both
native grouped GEMMs at ~220-230 GB/s, no dequantise or BF16 fallback), BF16 dense projections 11.8 ms, target LM
head 3.8 ms, draft LM head 3.9 ms, draft Markov fold (7 reads of a 103 MB matrix) 3.0 ms, draft layers 2.2 ms,
~1,000 small kernels 4.9 ms.

Bytes per c1 step: dense BF16 2.79 GB (48 layer applications), LM head 0.93 GB, draft ~1.65 GB, MoE u x 0.23 GB
where u = distinct experts touched per layer visit. Measured on the live server (counter in the verify-cap overlay):
u = 28.3 at 8 verify rows (8 independent top-8 draws would give 51.6), 19.2 at 4 rows, 8 at 1 row. Total ~13.4 GB per
step: the measured 60 ms is 1.08x the floor at 240 GB/s. Every kernel is already near the memory bandwidth, so only
fewer bytes per accepted token can make decode faster: verify fewer rows (MoE union), read fewer draft bytes,
read the dense weights in fewer bits (exactly), or accept more tokens per step (a better draft, which needs training
and is out of scope).

## Decode levers (all screens on Spark_05 unless noted)

`vscreen`: 16 greedy prompts dumped for a text comparison against the base, speedbench quick (30 x 160, c1, T=0,
effort low), thinking load c8 (16 x 2048, T=0.8, effort high); thinking c1 (4 x 2048) for the combined arms.

| arm | speedbench quick all (prose) | step ms | tokens/step | thinking c8 | thinking c1 | greedy text = base |
|---|---:|---:|---:|---:|---:|---:|
| B0 base | 40.68 (32.26) | 61.4 | 2.448 | 187.8 | - | - |
| B0b base, 2 h later | 40.42 (32.06) | 61.7 | 2.448 | 198.5 | 52.6 | 16/16 |
| verify cap fixed:8 + union counter (inert check) | 39.63 (31.41) | 63.0 | 2.448 | 187.7 | - | 16/16 |
| verify cap fixed:4 + union counter | 41.97 (35.40) | 52.9 | 2.210 | 174.9 | - | 15/16 |
| verify cap fixed:4, remap off (identity check) | - | - | - | - | - | 16/16 vs fixed:4 remap on |
| verify cap v2 fixed:8 fused remap (overhead) | 40.38 (32.01) | 61.9 | 2.448 | 190.9 | - | 16/16 |
| verify cap cost 0.30/0.25/0.15/0 | 46.39 (38.53) | 50.9 | 2.356 | 186.6 | - | 15/16 |
| verify cap fixed:4 only at bs 1 | 43.01 (36.40) | 51.6 | 2.210 | 185.6 | - | 15/16 |
| draft gemv + fold8 + head8 (Spark_06) | 41.96 (33.68) | 59.1 | 2.487 | 187.5 | - | 15/16 |
| draft + MXFP8 draft layers (Spark_06) | 42.44 (34.21) | 58.6 | 2.451 | 192.3 | - | 14/16 |
| MoE tiles (Spark_06) | 40.84 (32.39) | 61.2 | 2.448 | 186.5 | - | 16/16 |
| MoE tiles + QKVR GEMV (Spark_06) | 41.10 (32.56) | 60.7 | 2.448 | 190.8 | - | 16/16 |
| C1: cost 0.30 + draft (no mx8) + tiles + QKVR | 51.22 (42.73) | 46.5 | 2.409 | 189.6 | 57.6 | 15/16 |
| C2: C1 + MXFP8 draft layers | 51.90 (43.57) | 45.8 | 2.356 | 199.9 | 58.7 | 14/16 |
| C3: C2 with cost 0.40/0.30/0.20/0 | 52.14 (44.01) | 44.2 | 2.255 | 192.3 | 59.2 | 13/16 |
| C4: C2 with cost 0.20/0.15/0.10/0 | 50.95 (42.53) | 47.7 | 2.431 | 199.5 | 60.8 | 14/16 |
| C5: C2 with cost 0.50/0.40/0.25/0 | 51.42 (43.35) | 42.4 | 2.106 | - | - | 14/16 |
| C6: C2 + pico12 dense,lmhead + prefill qkvr_split, partial_bf16 | 54.95 (48.06) | 43.1 | 2.239 | 195.7 | 61.1 | 0/16 (diverges in the first 20-370 characters of the reasoning; see below) |

Reading the table:

- C6 changes the floating-point summation order of every dense projection (pico's GEMV accumulates in a different
  order than SGLang's Triton GEMV; its packed weights decode to the same BF16 bits), so every greedy text leaves the
  base's trajectory early, at a word choice inside the reasoning, and continues coherently. Quality is checked with
  the speedbench quality guard and the sanity set in the final runs (README).

- The base is deterministic across boots (B0 and B0b: identical greedy text, speed within 0.6 %).
- Greedy text identity is not the exactness gate for this model: any change of floating-point summation order or
  of the verify block boundaries moves near-tie tokens through the W4A4 activation quantisers (equally correct
  implementations agree on ~95 % of tokens teacher-forced). Exactness is established per lever: bit-identical
  kernel outputs in GPU checks (tiles, QKVR GEMV, prefill levers, packed BF16 == raw BF16), byte-identical text
  between remap on and off for the verify cap (so the dead rows never reach a live row), and the stopping-rule
  argument plus a Monte Carlo test for sampled requests.
- Thresholds: greedy speedbench peaks around 0.40, the sampled thinking c1 runs prefer lower thresholds (4 requests
  each, noisy). 0.30 is the default; `VERIFY_CAP` sets it.
- Offline replay of 18.7K logged c1 steps (`results/steplog/`, logged with a read-only step-log overlay and replayed offline):
  greedy fixed:8 37.6 -> best fixed width (5) 38.8 -> left-to-right cost policy 43.4 tok/s (+15 %; oracle 49.4);
  sampled 50.0 -> 53.3 (+6.6 %). The cost rule in serving (+14 % greedy) captures about what the best replayed
  policy does.

Rejected or parked:

- `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION=1`: flat (GPU-bound step).
- Fixed verify caps: lose on thinking at c8 (-7 % at fixed:4); the batch-aware cost rule replaced them.
- Hot-vocabulary draft: from 135K tokens of generated traces a 32K-token list covers only 92.7 % of held-out target
  tokens (94-95 % padded with low ids), and every miss is a forced rejection. Needs a multi-million-token trace
  corpus first.
- Exact 12-bit packed BF16 GEMV as a Triton port of Christopher Owen's packed format (`results/kbench/K3-*`):
  x1.11 on qkvr, x1.16 on the head, below the 1.15x kill line on qkvr; superseded by pico's CUDA kernel.
- `--speculative-num-draft-tokens` other than 8: refused by the trained draft (block size 8).

## Prefill

Where the time goes (one 8,192-token chunk at prefix 0, 2.40 s; composed from GB10 microbenchmarks of every kernel
class, `docs/prefill/`): MoE grouped GEMMs 13 %, MoE glue (activation quantise, SwiGLU-quantise, combine) 13 %,
useful dense GEMMs 10 %, the stacked Q/K/V projection's discarded rows 8 %, attention main loop 5 %, relative
position bias (built as `[t, 24, 1024]`, copied into FA4's sheared layout, then applied) 17 %, other elementwise work
34 %. The kernels themselves run near their peaks (cuBLAS BF16 88-92 TFLOP/s, NVFP4 grouped GEMM 164, FA4 without
bias 70, DRAM 224 GB/s); FA4 with Inkling's bias runs at 13 (sliding window) to 46 (global) TFLOP/s. Measured prefill
equals the ceiling that includes today's memory round trips and is 29-43 % of a GEMM+attention-only ideal.

Exact prefill levers (`overlay/inkling_prefill/`, all bit-identical to stock in the GPU check on real weights,
216/216 comparisons incl. the attention output and LSE for the bias lever):

| arm (speedbench longctx, cold, tok/s) | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| base (PF0, Spark_06) | 2,151 | 3,408 | 3,235 | 2,409 |
| qkvr_split + partial_bf16 (PFO2) | 2,336 | 3,781 | 3,558 | 2,537 |
| all four: + bias_layout + quant_once (PFO3) | **2,363** | **4,253** | **3,958** | **2,749** |
| `--chunked-prefill-size 16384` alone (PF1) | 2,160 | 3,399 | 3,310 | 2,493 |

- `qkvr_split`: the looped layers project three different inputs (q, k, v sources) through one stacked 8600x2304
  matrix as 3M rows and keep a third of the output; computing each source only against its own column block cuts
  that projection by 63 % (8K chunk: 10.55 -> 3.77 ms per call). Bit-identical from 128 rows (it is off below).
- `bias_layout`: one Triton kernel computes the relative-position bias directly in FA4's sheared layout.
- `quant_once`: each token's MoE input is quantised to NVFP4 once and gathered to its 8 experts, instead of 8 times.
- `partial_bf16`: the MoE combine writes BF16 directly.
- A suspected stock failure of SGLang's `route_metadata` above ~2,370 routed ids (shared-memory limit) shows up in a
  standalone call but not in serving: cold prompts of 328-774 tokens all answer on the stock server.
- Not adopted: chunk 16K (+2-3 % at 32K-128K, more activation memory), NVFP4 tile retune for prefill (<= 2 % of a
  GEMM, all configs bit-identical), prefill CUDA graphs off and 4K / 32K chunks were not screened (out of time).

## Final confirmation (both boxes)

The shipped configuration (`scripts/launch.sh` defaults) ran `bench/full.sh` once on each box
(`results/FINAL-s06/`, `results/FINAL-s05/`), against the stock engine's full runs on the same boxes
(`results/F2-sglang-base-s06/`, `results/F1-sglang-base-s05/`):

| | stock (06 / 05) | final (06 / 05) |
|---|---|---|
| speedbench full, all 73 | 40.16 / 41.09 | 55.51 / 56.06 |
| speedbench prose | 32.3 / 33.0 | 46.4 / 46.8 |
| step / tokens per step (c1) | 61.6 / 60.1 ms, 2.45 | 43.9 / 43.4 ms, 2.48 |
| decode table prose c1 / c4 / c16 | 32.8 / 74.4 / 181.5 ; 33.6 / 76.0 / 190.8 | 47.3 / 90.6 / 188.9 ; 47.8 / 89.6 / 192.3 |
| thinking c1 / c8 / c24 | 51.4 / 189.1 / 368.9 ; 53.3 / 198.1 / 380.9 | 63.9 / 192.8 / 371.8 ; 61.8 / 194.8 / 373.8 |
| cold prefill 1K / 8K / 32K / 128K | 2,137 / 3,399 / 3,230 / 2,426 ; 2,190 / 3,400 / 3,234 / 2,419 | 2,510 / 4,256 / 3,903 / 2,759 ; 2,474 / 4,275 / 3,979 / 2,836 |
| quality guard / sanity / needles 120K + 240K | (guard 30/30 on the base control) / 11/11 / PASS | 30/30 / 11/11 / PASS on both |
| MemAvailable after start, lowest | ~39-40 GiB | 39.8-40.1, lowest 39.1-39.3 GiB |

Where the remaining c1 time goes after the overlays (from the screens): the dense BF16 projections and LM head
(now 12-bit packed), the routed experts of the verified rows, and the draft. Doubling c1 speed would need
more accepted tokens per step (a better or longer-trained draft) or a lossy weight format; neither is in this recipe.
