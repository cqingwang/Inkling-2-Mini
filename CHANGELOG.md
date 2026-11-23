# Changelog

## v2.5 (2026-11-23)

One-pass checkpoint loading, a cheaper draft Markov walk and full-context draft attention by default. Median of
both boxes, `bench/full.sh` (`results/V25-s05/`, `results/V25-s06/`), against v2.4: speedbench full c1 64.2 (64.1)
at 2.383 tokens per step in both releases; decode prose c1 / c4 / c8 / c16 53.0 / 107.7 / 161.9 / 238.9
(52.9 / 107.3 / 159.6 / 236.4), code / json c16 260.4 / 277.1 (261.3 / 275.4); thinking c1 / c8 / c24
70.5 / 235.6 / 445.2 (68.2 / 239.9 / 439.4); cold prefill 1K / 8K / 32K / 128K 2,893 / 5,464 / 5,143 / 3,806
tok/s (2,939 / 5,456 / 5,148 / 3,805); decode at 32K / 128K 71.0 / 46.4 (63.3 / 48.2). At 200K / 250K on Spark_06,
c1 decode 43.8 / 37.9 tok/s, cold prefill 3,105 / 2,655 tok/s. Warm start-up 137 / 131 s and KV pool 1,105,792 /
1,106,816 global-layer tokens (Spark_05 / Spark_06); 16.5-16.6 GiB MemAvailable after start. The stress runs
reached 14.1 / 14.5 GiB lowest MemAvailable (Spark_05 / Spark_06) with thinking c24 at 310.7 / 309.2 tok/s alongside a
cold 240K-token needle, PASS. Quality guard
30/30, sanity 11/11, needles 120K / 240K PASS on both boxes.

- `inkling_draftbytes` `rnnhoist`: precompute the known 90 % of the Markov projection in FP32, run the 768x512
  dependent slice per step, then round once to BF16 and call the stock gates. Draft proposals only. The seven-step
  walk takes 320 / 454 / 596 us at 1 / 8 / 16 rows (584 / 661 / 782 us, cold L2, CUDA graph).
- `LOAD_NOMMAP=1`: SGLang's `--weight-loader-disable-mmap` reads the checkpoint in one pass. Warm start-up
  263-268 -> 131-137 s, target weight load 160 -> 22-34 s in the screen. `MALLOC_MMAP_THRESHOLD_=1048576`
  returns large frees to the OS before KV allocation, reducing the pool loss from 6 % to 2.6 % at the same fraction.
  `MEM_FRACTION` default 0.80 -> 0.81.
- `dwin:16384` is off by default again. The 16K window loses far-context copying. Controlled A/Bs on both boxes
  put net decode at -11 / -7 / -3 / +7 % at 32K / 64K / 128K / 250K. The option stays available for workloads
  mostly above 200K tokens.
- Screened, not adopted: Humming MoE, per-row INT8 draft weights, context n-gram proposals, extra draft vocabulary
  rows per step. Details and the same-harness vLLM comparison are in docs/experiments.md "v2.5".

## v2.4 (2026-11-22)

Sampled-path verification, a drafter attention window and a packed global-layer KV; all exact. Median of both boxes,
`bench/full.sh` (`results/V24-s05/`, `results/V24-s06/`), against v2.3: thinking c1 / c8 / c24 68.2 / 239.9 / 439.4
(69.8 / 236.5 / 427.9); decode prose c8 / c16 159.6 / 236.4 (156.5 / 232.9), code c16 261.3 (256.0), json c16 275.4
(268.3); decode at 128K context 48.2 (43.1, +12 %), at 32K the same step time as v2.4 with its levers off (47.2 vs 47.3 ms, same session; the full-run tok/s cell, 63.3 vs
69.6, follows the accepted tokens of the generated texts); 200K / 250K c1 decode 44.1 / 38.9
on Spark_06 (v2.3 38.1 / 35.1 on Spark_05); cold prefill 1K / 8K / 32K / 128K 2,939 / 5,456 / 5,148 / 3,805 tok/s
(2,914 / 5,352 / 5,036 / 3,751, run noise); speedbench full c1 64.1 (64.7) at identical tokens per step (2.383): a
same-session ablation on both boxes puts the c1 step with all v2.4 levers on and off within 0.6 %, so the -0.9 % is
run-to-run variation, not the levers; KV pool 1.11-1.12M tokens (-6 %: `MEM_FRACTION` 0.84 -> 0.80 for the 6.1 GiB
packed side pool), lowest MemAvailable 14.8-15.6 GiB under thinking c24 with a cold 240K-token prompt; quality guard
30/30, sanity 11/11, needles 120K / 240K PASS on both boxes.

- `inkling_vcap`: block verification (Sun et al., ICLR 2025) for sampled requests under the verify cap
  (`VCAP_BV=1` -> `--speculative-use-block-verification`, `INKLING_VERIFY_CAP_BV=mask`: dead drafts enter as a
  stopped draft); thinking c1 / c8 / c24 +3.0 / +4.8 / +4.0 % in the screen.
- `inkling_vcap` `VCAP_FASTPROBS=1`: the verify rule's max draft probabilities and the sampled accept's q from one
  read of the BF16 draft logits; +1.7 to +3.5 % (thinking c8, greedy c8 / c16).
- `inkling_vcap` `VCAP_FUSED=1`: sampled accept computed from the logits for >= 3 requests; code c16 +2 %.
- `inkling_draftbytes` `dwin:16384`: the draft's global layer reads a 128-token sink + the last 16,384 tokens; c1 step
  at 128K / 250K -1.4 to -2.4 / -4.0 to -4.8 ms (+2 to +6 %, both boxes), tokens per step unchanged.
- `inkling_kv12` (new, `KV12=1`, `KV12_SLOTS=2048`): full, committed global-layer KV pages also kept as exactly coded
  12-bit pages (2,048 slots per layer, 6.1 GiB) and decoded in the decode / verify attention; bit-identical partials;
  c1 step at 128K 62.2 -> 56.8 ms, 250K 80.1 -> 75.2 ms.
- `MEM_FRACTION` default 0.84 -> 0.80.
- Rejected: draft sampling temperature, 4-bit draft MLP (tokens-per-step-only gain within boot-to-boot variation, step
  time unchanged), late kill of doomed verify rows, shared-prefix KV reads, fused routed epilogue, streaming NVFP4 MoE
  kernel, centroid draft vocabulary, verify-policy retunes, 80K hot vocabulary. Details: docs/experiments.md "v2.4".

## v2.3 (2026-11-22)

Three exact levers. Median of both boxes, `bench/full.sh` (`results/V23-s05/`, `results/V23-s06/`), against v2.2:
speedbench full c1 64.7 (61.8, +5 %), prose 54.4 (51.9); decode prose c1 / c2 / c4 / c8 / c16 53.4 / 72.5 / 110.0 /
156.5 / 232.9 (51.0 / 69.6 / 107.0 / 154.1 / 228.8), code c8 178.3 (171.4); thinking c1 / c8 / c24 69.8 / 236.5 /
427.9 (67.1 / 219.6 / 419.6); prefill within run noise; KV pool 1.18-1.19M tokens (-2 %: +0.81 GB resident scales);
quality guard 30/30, sanity 11/11, needles PASS on both boxes.

- `inkling_scodes` (new, `SCODES=1`): NVFP4 expert block scales as row base + 5-bit codes, expanded inside a
  modified FlashInfer SM120 grouped-GEMM mainloop; -4.06 % of expert bytes, bit-identical outputs.
- `inkling_pico12` `stack1`: the stacked q/k/v of the looped layers in one pass over the coded weight (no kv pack).
- `inkling_draftbytes` `w8tile`: narrow-N tiles for the MXFP8 draft layers (bit-identical).
- Screened, not adopted: stream-K packed GEMM for 97-192 rows, prefill glue round 3 (both flat in serving).

## v2.2 (2026-11-22)

GB10 tile tables for the routed-expert GEMMs (`overlay/inkling_moe2`, default `MOE2=tiles`; bit-identical outputs).
Median of both boxes, `bench/full.sh` (`results/V22-s05/`, `results/V22-s06/`), against v2.1: decode prose c2 / c4 /
c8 / c16 69.6 / 107.0 / 154.1 / 228.8 (68.0 / 103.2 / 149.8 / 216.7), code c4 / c16 122.2 / 250.5 (115.8 / 240.3);
thinking c8 / c24 219.6 / 419.6 (220.8 / 405.4); c1 unchanged (speedbench 61.8, the c1 path does not use these tiles);
quality guard 30/30, sanity 11/11, needles PASS on both boxes. Same-day A/B on Spark_05 (control / moe2 / control):
decode +2 to +6 % at c2-c16. Prefill numbers move within run noise.

## v2.1 (2026-11-21)

Hot-vocabulary draft. Median of both boxes, `bench/full.sh` (`results/V21-s05/`, `results/V21-s06/`), against v2:
speedbench full c1 61.7 (58.1, +6 %), prose 51.7 (48.5), multilingual 42.3 (39.4); decode prose c1 / c4 / c16 51.0 /
103.2 / 216.7 (47.5 / 99.7 / 212.1), code c1 60.2 (56.7), json c8 / c16 173.6 / 265.6 (164.6 / 251.8); thinking c1 /
c8 / c24 67.8 / 220.8 / 405.4 (65.8 / 211.1 / 402.2); prefill and memory unchanged within run noise (1K / 8K / 32K /
128K 2,914 / 5,293 / 4,995 / 3,726 tok/s; 1.20M KV tokens, 17.5-17.8 GiB lowest MemAvailable); quality guard 30/30,
sanity 11/11, needles PASS on both boxes.

- `inkling_draftbytes`: default `hotvocab:65536` with the shipped list
  `overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json` (ranked on 1.95M tokens of the model's own outputs,
  `tools/hotvocab/`), and `wide8` (FP8 draft GEMMs read their weight once above 64 rows; bit-identical).
- Same-day A/B on both boxes: speedbench quick +6.0 / +6.1 %, greedy texts byte-identical 30/30.
- Rejected: KV page size 16 (flat decode; the fused prefill attention needs page 128).

## v2 (2026-11-21)

Five new overlay packages, a new verify-width rule and a larger KV pool; all levers exact (same weights, same target
distribution). Median of both boxes, `bench/full.sh`, against v1:

- speedbench full c1 58.1 (v1 55.8, +4 %); decode table prose c4 / c8 / c16 99.7 / 149.1 / 212.1 (90.1 / 121.5 /
  190.6: +11 / +23 / +11 %), code c4 / c8 / c16 +10 / +11 / +7 %; thinking c1 / c8 / c24 65.8 / 211.1 / 402.2
  (62.8 / 193.8 / 372.8: +5 / +9 / +8 %); cold prefill 1K / 8K / 32K / 128K 2,821 / 5,317 / 5,060 / 3,761 tok/s
  (+13 / +25 / +28 / +34 %); KV pool 1.19-1.21M global-layer tokens (794K, +52 %) with 17.5-17.9 GiB lowest
  MemAvailable; quality guard 30/30, sanity 11/11, needles 120K / 240K PASS.
- `inkling_vcap`: `policy:` mode, a calibrated left-to-right verify-width rule (per-position survival tables from 36K
  logged request-steps, one price for all batch sizes); default `policy:.../r2-theta004.json,bsmax=16`.
- `inkling_cert` (new): certified greedy LM-head argmax (6-bit screen + exact pico GEMV on the candidate tiles).
- `inkling_wide` (new): packed 12-bit GEMM for 17-256-row verify batches, decode / verify forwards only.
- `inkling_vattn` (new): verify attention on the split-KV relative-bias kernel up to 48 requests, bias in-kernel.
- `inkling_attn` (new): fused relative-bias prefill attention (sliding-window and global layers).
- `inkling_pfuse` (new): prefill glue fusion (loop-message mixes, attention-residual mix + norm, short conv + residual).
- `--mem-fraction-static` 0.62 -> 0.84 (worst case with thinking c24 and a 240K-token prompt measured at 0.72-0.86).
- Rejected: L2 prefetch / warm kernels before the small per-visit kernels, a native layer-stack backend (1.13x upper
  bound), the verify policy above 16 requests. Details: docs/experiments.md "Round 2 (v2)".

## v1 (2026-11-21)

- 2026-11-21: first release. SGLang `lmsysorg/sglang:inkling-2-mini` (`sha256:efac853b…`) with the cookbook's DGX
  Spark DSpark cell (`--mem-fraction-static 0.62`, 262K context, text only) plus five import-time overlays
  (`overlay/`), all on by default in `scripts/launch.sh` (`OVERLAY=0` for the stock engine):
  - `inkling_vcap`: verify-width cap with dead-row expert remap and a batch-aware threshold rule
    (`cost:1=0.30,2=0.25,4=0.15,8=0.0`);
  - `inkling_draftbytes`: draft logits on the GEMV, FP8 Markov fold, FP8 draft LM head, MXFP8 draft layers;
  - `inkling_kernels`: GB10 NVFP4 tile table at decode, stacked Q/K/V on one GEMV;
  - `inkling_pico12`: pico's exactly coded 12-bit BF16 GEMV for the dense projections and the LM head;
  - `inkling_prefill`: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once,
    BF16 MoE combine.
  Speed and quality numbers: README; every screen, the engine comparison and the rejected levers:
  docs/experiments.md.
- 2026-11-21: SGLang vs vLLM on the same benchmark, both engines on both test boxes: decode tied at c1 (within
  3 %), SGLang 1.3-3.4x faster cold prefill and twice the free host memory -> SGLang is the base.
- 2026-11-21: repository created. Weights revision `ed40e7da`, vLLM image `sha256:c2dc759f…` for the comparison.
  Launcher with a start-up memory guard, download script with sha256 checks, benchmarks (`bench/`: decode table,
  thinking load, sanity checks, screening and full runs).
