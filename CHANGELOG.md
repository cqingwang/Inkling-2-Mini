# Changelog

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
