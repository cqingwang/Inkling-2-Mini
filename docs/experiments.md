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

## Round 2 (v2)

Same boxes, same image, same harness. Every lever below was screened alone against a control booted the same day on
the same box (`results/screen-r2/`, one line per arm in `summary.jsonl`); the stack was then confirmed with
`bench/full.sh` on both boxes (`results/V2-s05/`, `results/V2-s06/`). Kernel checks, microbenchmarks and analysis
scripts are in `results/kbench/round2/<lever>/`.

### Where the time goes with the round-1 stack

Torch profile of the shipped round-1 stack (greedy prose, 40 steps each; `results/kbench/round2/residual/` has the
corrected analyser `analyze2.py` and per-class tables, every kernel attributed, GPU idle 0.4-0.6 ms per step):

| ms per step | c1 | c4 | c8 | c24 |
|---|---:|---:|---:|---:|
| step | 39.9 | 85.3 | 117.1 | 171.8 |
| routed experts (grouped GEMMs) | 16.5 | 53.0 | 75.8 | 98.8 |
| dense qkvr / wo (pico at <= 16 rows, cuBLAS above) | 9.5 | 12.9 | 14.8 | 16.7 |
| target LM head | 2.9 | 4.4 | 5.0 | 7.0 |
| verify attention (SWA + global) | 1.2 | 2.3 | 6.8 | 20.6 |
| draft (layers, FP8 head, FP8 fold) | 5.2 | 5.8 | 6.7 | 11.4 |
| norms, attn-res, short conv, loop controllers, MoE glue | 2.9 | 4.4 | 6.0 | 13.9 |

Measured expert unions (stock engine with the read-only counter, 36K logged request-steps,
`results/screen-r2/R2SL-s06/`): per MoE visit U(1) / U(4) / U(8) rows per request = 32.8 / 56.7 / 68.8 at 8 requests
and 51.4 / 79.0 / 91.2 at 16 requests, far from the 128 experts. The round-1 rule verified all 8 rows from 5 requests
up because the union was assumed saturated; it is not.

### Decode levers (screens on Spark_05, same day)

Controls: C0a and C0b (round-1 stack, booted before and after the arms) agree within 1-3 % except the sampled thinking
cells (+-5 %).

| arm | sbquick c1 | thinking c4 / c8 / c24 | prose c4 / c8 / c16 | code c4 / c8 / c16 |
|---|---:|---|---|---|
| C0a control | 54.93 | 126.0 / 196.2 / 374.3 | 92.1 / 121.7 / 195.3 | 103.1 / 148.7 / 225.2 |
| C0b control | 54.97 | 131.4 / 197.8 / 373.4 | 91.8 / 120.2 / 192.1 | 102.8 / 143.6 / 224.7 |
| V1a policy theta 0.04 | 55.58 | 130.8 / 200.3 / 346.1 | 96.9 / 137.4 / 205.3 | 105.4 / 157.0 / 228.6 |
| V1b policy theta 0.04 (repeat) | 55.46 | 133.6 / 188.9 / 360.3 | 96.1 / 135.6 / 202.8 | 105.0 / 152.4 / 230.2 |
| V2a cost:1=0.30,24=0.25 | 55.01 | 133.8 / 191.0 / 359.2 | 94.3 / 134.3 / 202.2 | 107.0 / 152.1 / 233.5 |
| V3a policy, bsmax=12 | 55.55 | 143.5 / 196.9 / 372.2 | 96.4 / 138.8 / 192.1 | - / 154.7 / 231.3 |
| CERT certified greedy head | 56.70 (56.65) | - | 90.2 / 122.9 / 194.5 | 107.1 / 146.2 / 227.9 |
| VAT verify attention | 55.07 | - / 195.9 / 398.0 | 92.1 / 125.8 / 204.0 | 102.8 / 142.7 / 235.2 |
| WID wide GEMM (all forwards) | 54.41 | - / 213.5 / 367.0 | 95.8 / 126.4 / 197.3 | 106.9 / 145.3 / 227.9 |

- **Verify policy** (`inkling_vcap` `policy:` mode). Offline replay on the logged steps (`results/kbench/round2/policy/`)
  priced a calibrated left-to-right rule with one price for all batch sizes at +0.5 / +2 / +5 / +6 % at 1 / 4 / 8 / 16
  requests. Live: greedy decode +5 / +13 / +5 % at c4 / c8 / c16, thinking c4 / c8 +4 / +2 % on average; at 24
  thinking requests both policy runs were below both controls (346 / 360 vs 374 / 373), the bsmax=12 run equal
  (372). Shipped with `bsmax=16`, which keeps the c16 greedy gain and verifies all rows above 16 requests.
- **Certified head**. Serving check mode first (`R2CHK`): 5,670 certified steps, 45,584 rows, 0 mismatches against
  the full head; 216 candidate tiles of 12,564 per step on average. Then the A/B: +3.1 % at c1, nothing above 16 rows
  (not eligible).
- **Verify attention**. Above 32 verify rows the stock path padded q and the bias to 4 heads per KV group, sheared the
  bias to [t, 32, 1280] and ran FA4; the split-KV kernel used at <= 4 requests has no batch-size dependence. The cap
  alone (32 -> 384 rows) gives most of the gain, the in-kernel bias the rest (visit 1.4-1.7x faster at 8-48
  requests). Long-context c1-c4 verify is already at the DRAM floor (222-235 GB/s at 32K-128K).
- **Wide GEMM**. In this screen it also took 17-256-row prefill chunks of short prompts (the c1 text changed:
  2.304 tokens per step vs 2.272 in every other arm); it ships restricted to decode / verify forwards
  (`INKLING_WIDE_SCOPE=verify`), confirmed in the full runs (c1 decode unchanged).

### Prefill levers (Spark_05, same day, speedbench longctx, cold)

| arm | 1K | 8K | 32K | 128K | guard / sanity / needles |
|---|---:|---:|---:|---:|---|
| C0b control (round-1 prefill levers) | 2,522 | 4,265 | 3,972 | 2,805 | - |
| ATT fused relative-bias attention | 2,694 | 4,887 | 4,669 | 3,535 | 30/30, 11/11, 120K + 240K PASS |
| PFA attention + glue fusion | 2,881 | 5,487 | 5,179 | 3,818 | 30/30, 11/11, 120K + 240K PASS |

The 240K needle's prefill took 89 s with ATT vs 133 s on the control. Decode tok/s at 32K / 128K moved in these arms
(68 / 45 control, 56-62 / 39-43 with the levers), but the time per verify step did not (32K 54-56 ms, 128K 66-69 ms in
all three): the two decode documents produce different text when the prefill numerics change, and the accepted tokens
per step range from 2.2 to 5.3 between documents. Prefill budget per 8K chunk before these levers (rebuilt with the
measured 164 TFLOP/s NVFP4 rate, `results/kbench/round2/residual/out/prefill.txt`): elementwise remainder 744 ms
(loop-message mixes 233, attention-residual mixes 143, norms 74, ...), routed-expert GEMMs 317, dense 250, MoE glue 211,
relative bias 219, attention main loop 119.

### KV pool (Spark_06, MemAvailable logged every 5 s from container start)

Worst case per setting: thinking load c24 (48 x 2,048 tokens) together with a cold 240,104-token needle prompt, then
40 min of 28 concurrent mixed requests.

| `MEM_FRACTION` | global-KV tokens | MemAvailable after start | lowest | needle 240K |
|---:|---:|---:|---:|---|
| 0.62 (round 1) | 794K | ~40 GiB | 39.1 GiB (full run) | PASS |
| 0.72 | 985,088 | 29.4 GiB | 26.9 GiB | PASS |
| 0.80 | 1,132,800 | 22.1 GiB | 18.5 GiB | PASS |
| 0.86 | 1,257,856 | 15.6 GiB | 12.1 GiB | PASS |

0.84 is the default: 1.19-1.21M tokens and 17.5-17.9 GiB lowest during the full runs, ~14 GiB expected in the worst
case above. The start-up peak is not lower than the steady state at any setting.

### Not adopted in round 2

- Loop-controller / small-kernel launch fusion and L2 warming: SGLang PR #51926 (fused loop controllers) is already in
  the image. An overlay that issues `prefetch.L2` for the next projection's weights (exact, bit-identical outputs) saves
  nothing: GB10 drops most of these instructions (an 8 MB read takes 45 us cold, ~40 us after an L2 prefetch, 14 us
  after a normal read). A side-stream read-and-discard warm before the consumer moves the same DRAM bytes and contends
  with its tail: -0.13 ms per step at c1, 0 at c24.
- A native layer-stack backend (SGLang keeps the API, scheduler and KV; the backend owns the 48-visit core): after
  crediting the round-2 overlay levers, removing every boundary-specific cost (cross-op fusion to the weight-byte floor,
  launch chaining, eager glue) gives at most 1.13x geometric mean at c4 / c8 / c24 (1.14x at c1), below the 1.20x we
  require before building it (`results/kbench/round2/residual/out/budget.txt`).
- The verify policy above 16 requests (above).
- Long-context decode beyond the DRAM floor: the remaining lever is an FP8 KV cache (~+18-20 % at 128K by bytes), which
  changes the numerics and is outside this recipe's exact default.

### Confirmation (both boxes, `bench/full.sh`)

| | round 1 (06 / 05) | v2 (05 / 06) |
|---|---|---|
| speedbench full, all 73 | 55.51 / 56.06 | 58.19 / 58.10 |
| decode table prose c1 / c4 / c8 / c16 | 47.3 / 90.6 / 120.3 / 188.9 ; 47.8 / 89.6 / 122.8 / 192.3 | 47.5 / 98.7 / 148.1 / 213.2 ; 47.5 / 100.6 / 150.2 / 211.0 |
| thinking c1 / c8 / c24 | 63.9 / 192.8 / 371.8 ; 61.8 / 194.8 / 373.8 | 63.5 / 210.7 / 406.7 ; 68.2 / 211.5 / 397.7 |
| cold prefill 1K / 8K / 32K / 128K | 2,510 / 4,256 / 3,903 / 2,759 ; 2,474 / 4,275 / 3,979 / 2,836 | 2,833 / 5,393 / 5,115 / 3,785 ; 2,809 / 5,241 / 5,004 / 3,737 |
| KV pool / MemAvailable lowest | 793,728 / 39.1-39.3 GiB | 1,206,016 ; 1,187,840 / 17.5-17.9 GiB |
| quality guard / sanity / needles 120K + 240K | 30/30 / 11/11 / PASS | 30/30 / 11/11 / PASS on both |

## v2.1: hot-vocabulary draft

The draft's vocabulary-sized reads (draft LM head in FP8 and seven Markov fold rows per step) cost 3.5-8 ms per step
at every batch size in the v2 profile. `hotvocab:N` (round-1 code in `inkling_draftbytes`) restricts the draft to the
first N ids of a ranked list; every target token outside the list is a forced rejection at that position, so the list
must cover the model's real output.

Corpus (`tools/hotvocab/`): 46,479 prompts from 14 open datasets (chat 47 %, code 16 %, math 13 %, tool use 9 %,
structured output 9 %, summaries 7 %; 22 % non-English), speedbench prompts excluded by n-gram containment; 2,311 of
them generated on the recipe's own server (28 concurrent, mixed effort and temperature, exact output token ids):
1.95M tokens. Held-out coverage (split by prompt, `results/kbench/round2/hotvocab/cov19m.txt`):

| N | 32K | 48K | 64K | 96K |
|---|---:|---:|---:|---:|
| all held-out | 94.4 | 97.5 | 98.8 | 99.8 |
| English / Polish / German / Chinese | 96.0 / 90.9 / 86.0 / 91.5 | 98.3 / 96.6 / 92.9 / 96.8 | 99.3 / 98.6 / 96.9 / 98.7 | 99.9 / 99.8 / 99.6 / 99.7 |
| speedbench greedy texts (never ranked on) | 98.0 | 99.1 | 99.6 | 99.8 |

The modelled c1 optimum is flat between 48K and 80K (+5.4 % at 64K); 64K was screened. Same-day A/B
(`results/screen-r2/V21H-*`, `V2C-*`; v2 + `hotvocab:65536,wide8` vs v2):

| box | sbquick | prose c1 / c4 / c8 / c16 | thinking c1 / c4 / c8 / c24 | greedy texts identical |
|---|---|---|---|---|
| Spark_05 | 61.08 vs 57.64 | 51.0 / 103.2 / 150.5 / 219.3 vs 47.7 / 100.2 / 145.6 / 216.2 | 68.2 / 157.1 / 220.5 / 408.4 vs 62.3 / 146.9 / 212.4 / 388.8 | 30 / 30 |
| Spark_06 | 60.70 vs 57.20 | 50.6 / 102.2 / 149.5 / 218.1 vs 47.4 / 101.2 / 150.4 / 212.7 | 66.8 / 143.3 / 214.7 / 399.7 vs 65.5 / 143.0 / 205.9 / 292.1 (control c24 outlier) | 30 / 30 |

Speedbench quick c1 step 37.5 vs 39.8 ms at equal tokens per step (2.14 median in both; multilingual 1.74 vs 1.73).
`wide8` (in the same arm) only changes calls above 64 rows (c9+ draft head) and is bit-identical to the previous
kernel. Confirmed with `bench/full.sh` on both boxes (CHANGELOG v2.1).

KV page size 16 (thefullctx measured +19 % decode at 15.6K context on their stack), on v2, Spark_06: speedbench quick
57.05 vs 57.20, thinking c8 211.6, decode at 32K / 128K 66.0 / 42.5 (no gain), cold prefill 8K 4,210 vs ~5,300 (the
fused prefill attention covers page 128 only and falls back). Not adopted.

## v2.2: routed-expert GEMM tiles at verify batch sizes

Round 1 tuned the NVFP4 tiles for decode on evenly spread synthetic routing. Replayed with real top-8 routing (16
greedy speedbench texts through a CPU reference of the model, verify blocks of B requests x 8 positions, the verify
cap's dead-row remap applied; unions 52 / 74 / 103 at 4 / 8 / 16 requests, close to the served 57 / 69 / 91), cold L2,
real layer-0 NVFP4 and layer-20 MXFP8 weights: all 12 NVFP4 tile configurations and all 6 MXFP8 tactics give outputs
bit-identical to the shipped choice. The fastest ones: NVFP4 `swap_ab=1, tile_n=32, tile_k=256` at 9-128 tokens per
call (w13 +3.4 to +8 %), MXFP8 CuTe tactic `tile_m=32, tile_n=128` at 512-1536 routed rows (w2 +2 to +19 %). MoE time
per verify step: c4 55.6 -> 54.1, c8 77.8 -> 75.0, c16 112.9 -> 105.8, c24 125.0 -> 123.4 ms
(`results/kbench/round2/moe/`). A multi-row SwiGLU-quantize kernel (bit-identical) saved only 0.13 ms at c24 and was
slower below; not adopted.

Same-day A/B on Spark_05 (`results/screen-r2/R3C-s05`, `R3M-s05`, `R3Cb-s05`; v2.1 control / + moe2 / control):

| | control (mean of 2) | moe2 |
|---|---|---|
| decode prose c2 / c4 / c8 / c16 | 68.1 / 101.2 / 150.1 / 219.0 | 69.8 / 107.1 / 153.1 / 231.7 |
| decode code c2 / c4 / c8 / c16 | 80.9 / 114.7 / 167.7 / 242.3 | 83.7 / 119.2 / 168.0 / 255.4 |
| decode json c4 / c8 / c16 | 121.9 / 175.0 / 260.7 | 128.0 / 181.3 / 272.2 |
| thinking c4 / c8 / c24 | 141.4 / 213.6 / 411.8 | 141.1 / 216.3 / 418.7 |
| speedbench quick c1 | 61.07 | 60.93 |

Confirmed with full runs on both boxes (CHANGELOG v2.2). During the Spark_06 full run an unrelated CPU compile job
ran on the host for 8 minutes (thinking c24 and long-context cells; MemAvailable down to 4.05 GiB); those cells agree
with Spark_05 within run noise and every check passed, but they are the less clean of the two.
