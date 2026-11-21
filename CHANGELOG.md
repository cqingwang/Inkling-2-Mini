# Changelog

## Unreleased

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
