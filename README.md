# Inkling-2-Mini on one NVIDIA DGX Spark

Serving [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on a single DGX Spark
(GB10, sm_121, 121.6 GiB unified memory) with the model's own NVFP4 / MXFP8 weights and its bundled DSpark draft,
on SGLang, plus a set of import-time speed overlays that leave the model's outputs as they are.

**Status (2026-11-21):** one profile. SGLang `lmsysorg/sglang:inkling-2-mini` with the cookbook's DGX Spark DSpark
settings, and five overlay packages (`overlay/`) that cut the bytes read per decode step and the work per prefill
chunk. Against the same engine without the overlays: speedbench +37 % (prose +43 %) at c1, cold prefill +15 to +25 %.
No requantisation, no training, no change of the target distribution; `OVERLAY=0` runs the stock engine.

| | stock SGLang cookbook cell (mem 0.62) | this recipe |
|---|---:|---:|
| speedbench full, all 73 prompts, c1 | 40.6 | **55.8** |
| speedbench prose, c1 | 32.6 | **46.6** |
| thinking load c1 (effort high, T=0.8) | 52.3 | **62.8** |
| cold prefill, 8K / 128K tokens, tok/s | 3,399 / 2,422 | **4,266 / 2,797** |

## Speed

Measured 2026-11-21 on one Spark per run, GPU clock capped at 2200 MHz by the host's own timer, fresh start of the
container, MemAvailable logged every 5 s. Final configuration on Spark_06 (`results/FINAL-s06/`) and repeated on
Spark_05 (`results/FINAL-s05/`); the tables show the median of the two boxes (each cell is one run per box); the stock engine was measured the same way on both boxes (`results/F1-sglang-base-s05/`, `results/F2-sglang-base-s06/`). 

### speedbench

[speedbench](https://github.com/knapcio/speedbench) 0.3 (`6733653`), full mode: 73 frozen prompts in six categories,
c1, temperature 0, 256 output tokens, reasoning effort low (the model always thinks; its reasoning tokens count as
decode tokens). Decode tok/s, geometric mean per category:

| prose | code | structured | agentic | multilingual | long context | all 73 prompts |
|---:|---:|---:|---:|---:|---:|---:|
| **46.6** | 61.5 | 59.1 | 71.2 | 38.2 | 64.2 | **55.8** |

Stock engine on the same boxes: prose 32.6, code 48.0, structured 44.3, agentic 49.3, multilingual 25.3, long
context 51.0, all 40.6. Per box: 55.5 / 56.1 (stock 40.2 / 41.1). 73 of 73 prompts valid in every run; median time
to first token 174 ms (stock 184 ms); 2.48 accepted tokens per verify step (stock 2.45) at a 43.9 ms step (stock
60.9 ms).

### Decode by prompt type and concurrency

`bench/decode_table.py`: speedbench's prompts split by type (prose, code, structured, JSON), temperature 0, 256 new
tokens, reasoning effort none (the model still writes a short thought). c1 is the decode speed of one stream (median
of three requests); above c1 the aggregate over the level's wall time, per-stream decode in brackets.

| prompt type | c1 | c2 | c4 | c8 | c16 |
|---|---:|---:|---:|---:|---:|
| prose | **47.5** | 66.3 (34.8) | 90.1 (23.5) | 121.5 (16.1) | 190.6 (12.8) |
| code | 56.4 | 73.2 (39.5) | 102.8 (28.9) | 147.2 (19.9) | 220.8 (15.7) |
| structured | 56.1 | 70.4 (36.4) | 103.1 (27.1) | 146.6 (19.4) | 230.9 (15.7) |
| json | 60.3 | 76.5 (40.4) | 106.5 (28.6) | 147.8 (20.2) | 241.1 (16.7) |

Stock engine, same protocol: prose 33.2 / 51.0 / 75.2 / 117.0 / 186.2, code 43.6 / 66.8 / 87.2 / 134.8 / 221.3,
structured 49.7 / 66.4 / 89.9 / 141.8 / 225.1, json 46.5 / 58.8 / 89.3 / 141.9 / 232.2. The gain is largest for one
to four streams; from five concurrent requests the verify cap stops cutting (the 8 rows of several requests already
touch most experts) and c8 / c16 are within run noise of the stock engine.

### Thinking workloads

`bench/think_load.py`: seeded arithmetic puzzles, reasoning effort high, temperature 0.8, 2,048 (c1) or 4,096 tokens
per request, aggregate completion tok/s (median per-request decode in brackets). This is the shape of an agent
workload: long reasoning, many requests at once.

| c1 | c8 | c24 |
|---:|---:|---:|
| **62.8** (64.0) | 193.8 (28.9) | 372.8 (18.2) |

Stock engine: c1 52.3, c8 193.6, c24 374.9. One run per box per cell (sampled); c8 and c24 vary ~5 % from run to run.

### Prefill and long context

speedbench `longctx`: a generated document with a unique salt (no prefix-cache hit) plus a question; cold TTFT,
then the same document with another question (served from the prefix cache); decode at 32K context.

| | 1K | 8K | 32K | 128K |
|---|---:|---:|---:|---:|
| prompt tokens | 1,008 | 8,160 | 32,703 | 130,921 |
| TTFT cold | 0.40 s | 1.91 s | 8.30 s | 46.8 s |
| **prefill, cold, tok/s** | **2,492** | **4,266** | **3,941** | **2,797** |
| TTFT, document cached | 0.24 s | 0.24 s | 0.24 s | 23.5 s |
| decode tok/s at that context | - | - | 62.5 | 42.0 |

Stock engine: 2,163 / 3,399 / 3,232 / 2,422 tok/s cold (TTFT 0.47 / 2.40 / 10.1 / 54.0 s), decode 55.1 at 32K and
37.1 at 128K. The 128K cached TTFT is the median of one cache hit (~0.3 s) and one miss (the second 128K document
evicts sliding-window state of the first), the same as the stock engine.

### Memory and start-up

| | |
|---|---|
| MemAvailable after start / lowest during the whole run | 39.8-40.1 GiB / 39.1-39.3 GiB |
| KV pool | 793,728 tokens (global layers) + 79,360 (sliding window) + 715 conv-state slots |
| start-up to the first answer | 223-238 s with warm caches; ~6-7 min the first time (JIT, CUDA-graph capture, pico12 extension build) |

Details in "Memory budget" below.

### Quality checks

Same runs, same checks as for the stock engine (`bench/full.sh`):

- speedbench quality guard (30 exact-answer items, T=0): **30 / 30** on both boxes (stock engine: 30 / 30).
- `bench/sanity.py`: 11 / 11 on both boxes: two chats, a reasoning puzzle at effort none / low / medium / high
  (all correct; effort reaches the template, top-level `reasoning_effort` and `chat_template_kwargs` both work),
  a tool call and the answer after the tool result, a JSON-schema constrained answer.
- Needle (passphrase at 50 % depth of varied filler): **PASS** at 120,097 tokens (45 s) and 240,104 tokens
  (114-127 s) on both boxes.
- What "exact" means here: no lever changes the model's weights or its output distribution. The verify cap only
  decides how many draft tokens to verify (greedy output is still the target's argmax; for sampled requests the
  width is a left-to-right stopping rule on the draft's own probabilities, Monte Carlo tested); the draft levers
  only change proposals; the kernel levers compute the same math, bit-identical where stated, otherwise with a
  different floating-point summation order (pico's GEMV; its packed weights decode to the same BF16 bits). Any
  such change moves near-tie tokens on this model (its W4A4 activation quantisers amplify last-bit differences;
  equally correct implementations agree on ~95 % of tokens teacher-forced), so greedy texts are not byte-identical
  to the stock engine; the stock engine itself is deterministic from boot to boot.

## What the overlays do

Decode at c1 is bound by memory bandwidth: a step reads ~13.4 GB (BF16 dense projections 2.8 GB over 48 layer
applications, LM head 0.93 GB, draft ~1.65 GB, and the routed experts touched by the 8 verify rows, ~6.5 GB) and
the stock engine runs at 1.08x the 240 GB/s floor with the GPU 99 % busy ([profile](docs/profile/PROFILE.md)).
So the overlays cut bytes, not overheads. Prefill is the opposite: the kernels run near their peaks, but
intermediates make round trips through memory and the stacked Q/K/V projection computes three times the rows it
keeps ([prefill analysis](docs/prefill/PREFILL.md)).

Every lever is switched by an environment variable of `scripts/launch.sh` and implemented as an import-time patch in
`overlay/` (no engine file is edited; `OVERLAY=0` gives the stock cookbook engine). Effects are single-run screens
against the same-box base (speedbench quick: 30 prompts x 160 tokens, c1, T=0); the full stack is confirmed in the
tables above. Raw files: `results/screen/`; details and rejected levers: [docs/experiments.md](docs/experiments.md).

| lever | switch (default) | effect, alone | output | credit |
|---|---|---|---|---|
| Verify-width cap with dead-row expert remap: per request and step, verify only the first L of the 8 DSpark rows, L from the draft's own per-position confidence (left-to-right survival >= a per-batch-size threshold); dead rows reuse the anchor row's experts, so the MoE reads fewer distinct experts (28.3 -> 19.2 per visit at 4 rows) | `VERIFY_CAP=cost:1=0.30,2=0.25,4=0.15,8=0.0` | speedbench quick 40.7 -> 46.4 (+14 %), prose +19 %, c1 step 61.4 -> 50.9 ms; c8 thinking unchanged (threshold 0 from 5 requests up) | target distribution unchanged | ours (idea from our DeepSeek-V4.1 verify cap; batch-aware cost rule after our GLM-5.3-Flash pay-for-rows rule) |
| Draft logits through the bandwidth-efficient GEMV, FP8 copy of the draft's Markov fold matrix and of the LM head for the draft only, MXFP8 copies of the 3 draft layers | `DRAFT_BYTES=gemv,fold8,head8,draftmx8` | 40.7 -> 42.4, step -2.8 ms, acceptance unchanged (2.45-2.49 tokens/step) | draft proposals only | ours; FP8 draft head after our DeepSeek-V4.1 `DSV41_DRAFT_HEAD_FP4` |
| NVFP4 expert GEMM tiles for GB10 at decode (`tile_k` 256 instead of the RTX 5090 table's 128) | `KERNELS=moe_tiles` | MoE GEMM +5-7 % (microbench), step -0.2 to -0.5 ms | bit-identical (bench + end-to-end text) | ours; FlashInfer / CUTLASS kernels |
| Stacked Q/K/V projection of the looped layers on one Triton GEMV instead of cuBLAS at <= 16 rows per source | `KERNELS=qkvr_gemv` | 187 -> 173 us per call, step -0.4 ms | bit-identical in the GPU check | ours; SGLang's sm12x GEMV kernel |
| pico's exactly coded 12-bit BF16 GEMV (`bf16x`) for qkvr / wo and the target LM head at <= 16 rows | `PICO12=dense,lmhead` | qkvr 130 vs 174 us, wo 52 vs 66 us, LM head 2.90 vs 3.96 ms per call (M=8, cold); with the stack, step 45.8 -> 43.1 ms, speedbench quick 51.9 -> 55.0 | weights decode to the same BF16 bits; FP32 summation order differs from SGLang's GEMV | **Andy Shu (@blaesus)**, [dsgb10](https://github.com/blaesus/dsgb10) / [pico](https://github.com/Picom-code/pico-dgx-ds4) (MIT), vendored unmodified; integration ours |
| Prefill: stacked qkvr projection computed per source column block (no wasted 3x rows) | `PREFILL=qkvr_split` | that projection -63 % at 8K chunks | bit-identical at >= 128 rows | ours (found by our prefill roofline) |
| Prefill: relative-position bias written directly in FA4's sheared layout | `PREFILL=bias_layout` | with quant_once: 8K prefill 3,781 -> 4,253 tok/s | bit-identical (attention output and LSE) | ours |
| Prefill: MoE input quantised to NVFP4 once per token, then gathered to its 8 experts | `PREFILL=quant_once` | (with bias_layout, above) | bit-identical | ours |
| Prefill: MoE combine writes BF16 directly | `PREFILL=partial_bf16` | combine -22 to -32 %; with qkvr_split: 8K 3,408 -> 3,781 tok/s | bit-identical | ours |
| SGLang DSpark cookbook cell (FA4, page 128, 8 draft tokens, 4 `SGLANG_OPT_*` switches) | always | base | - | SGLang |
| `--mem-fraction-static 0.62` | `MEM_FRACTION` | 794K KV tokens with the overlays (843K without), ~40 GiB host memory left | - | ours |

Tried and not adopted (details in [docs/experiments.md](docs/experiments.md)): vLLM (same decode, slower prefill,
half the free memory), `SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION` (flat), fixed verify caps (lose on concurrent
thinking), a hot-vocabulary draft (not enough trace data for 99 % coverage), the Triton port of a packed-BF16 GEMV
(superseded by pico's kernel), NVFP4 prefill tile retunes (<= 2 %), other draft block sizes (refused by the draft).

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
scripts/launch.sh up                      # SGLang + DSpark + overlays on 127.0.0.1:8000; ~7 min cold, ~4 min warm
#   or: OVERLAY=0 scripts/launch.sh up    # the same engine without the overlays (stock cookbook cell)
scripts/launch.sh status                  # container, API, MemAvailable
python3 bench/sanity.py all               # chat, effort ladder, tool call, JSON schema, 120K needle
scripts/launch.sh down
```

The API is OpenAI-compatible at `http://127.0.0.1:8000/v1`, model name `inkling-2-mini`. The server listens on
loopback only; from another machine use `ssh -L 8000:127.0.0.1:8000 <spark>`, or set `HOST=0.0.0.0` on a trusted LAN.
Do not forward the port through your router.

The first start builds the pico12 CUDA extension (~30 s) and compiles the overlay Triton kernels; both are cached
in `~/.cache/inkling-2-mini-sglang` like SGLang's own JIT caches.

Do not start the server while a large download or copy is still running: on GB10 the page cache and the GPU share
the same memory (the "page-cache trap", see Credits). `scripts/launch.sh up` refuses to start next to another model
server, and starts `scripts/memguard.sh`, which stops the container if MemAvailable stays below 4 GiB during the first
20 minutes (an out-of-memory on GB10 can hang the box until a power cycle).

### Using it

Reasoning effort goes into the chat template, either as the top-level `reasoning_effort` field or as
`chat_template_kwargs: {"reasoning_effort": ...}`; both reach the template. Values: `none` (0.0), `minimal` (0.1),
`low` (0.2), `medium` (0.7), `high` (0.9, the default), `xhigh` / `max` (0.99), or a number in [0.0, 0.99]. The model
always writes a reasoning block, even at `none`; the reasoning parser returns it as `reasoning_content`, the answer as
`content`. Tools: OpenAI `tools` / `tool_calls`. Structured output: `response_format` with a JSON schema (xgrammar).

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
  -e INKLING_VERIFY_CAP=cost:1=0.30,2=0.25,4=0.15,8=0.0 -e INKLING_VERIFY_CAP_LOG_EVERY=0 \
  -e INKLING_DRAFT_BYTES=gemv,fold8,head8,draftmx8 \
  -e INKLING_KERNELS=moe_tiles,qkvr_gemv -e INKLING_MOE_TILES='1@2304:swap_ab=1,tile_n=32,tile_k=256;...' \
  -e INKLING_PICO12=dense,lmhead \
  -e INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16 -e INKLING_PREFILL_QKVR_OUT=strided \
  -e MAX_JOBS=4 \
  -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 \
  -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1 -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1 \
  --entrypoint python3 lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de \
  -m sglang.launch_server --model-path /model --served-model-name inkling-2-mini \
  --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128 \
  --speculative-algorithm DSPARK --speculative-num-draft-tokens 8 \
  --mem-fraction-static 0.62 --context-length 262144 \
  --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2 \
  --enable-metrics --host 127.0.0.1 --port 8000
```

Engine flags (the SGLang cookbook's DGX Spark DSpark cell, text only):

| flag / variable | why |
|---|---|
| `lmsysorg/sglang@sha256:efac853b…` | the `inkling-2-mini` tag pinned by digest (multi-arch; the Spark pulls arm64 `sha256:bc811ac7…`); SGLang `0.0.0.dev1+g03fde9cc3` |
| `--gpus all --ipc=host --shm-size 16g` | the GB10, and shared memory for the scheduler / detokenizer processes |
| `--network host` + `--host 127.0.0.1` | the API listens on loopback only |
| `--restart no` | nothing restarts behind your back after an out-of-memory |
| `-v …:/model:ro`, `HF_HUB_OFFLINE=1` | local weights at the pinned revision; no Hugging Face access at start |
| `-v ~/.cache/inkling-2-mini-sglang:/root/.cache` | keeps the JIT / Triton / FlashInfer caches and the pico12 extension build: a warm start takes ~4 min instead of ~6 |
| `MAX_JOBS=4` | caps parallel JIT compile jobs during the first start (cookbook) |
| `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1` | DSpark verify attention through the split-KV relative-bias decode kernel (cookbook) |
| `SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1`, `…_PREPARE_INPUTS_ONCE=1`, `…_SKIP_PREFILLED_INPUTS=1` | DSpark step glue inside the CUDA graphs (cookbook) |
| `--tp 1 --ep 1` | one GPU |
| `--attention-backend fa4` | FlashAttention-4 path with Inkling's relative bias (the only backend Inkling2 accepts) |
| `--dtype bfloat16` | activations; the weights stay NVFP4 / MXFP8 as shipped |
| `--page-size 128` | KV page size (cookbook) |
| `--speculative-algorithm DSPARK --speculative-num-draft-tokens 8` | the bundled 3-layer DSpark draft; the block size is fixed at 8 by the trained head |
| `--mem-fraction-static 0.62` | static pool for weights + KV + graphs: 794K KV tokens and ~40 GiB MemAvailable after start (see Memory budget) |
| `--context-length 262144` | per-request limit; the KV pool is shared, so this reserves nothing per request. The model supports 1,048,576 |
| `--grammar-backend xgrammar` | JSON-schema / regex constrained output |
| `--reasoning-parser inkling`, `--tool-call-parser inkling2` | thinking to `reasoning_content`, answer to `content`; OpenAI `tool_calls` |
| `--enable-metrics` | Prometheus `/metrics` (speedbench reads them) |

Overlay switches (`OVERLAY=0` turns all of them off; each is described in `overlay/<package>/README.md`):

| launch.sh variable (container env) | default | what |
|---|---|---|
| `OVERLAY_DIR` (mounted at `/opt/inkling-overlay`, on `PYTHONPATH`) | `overlay/` of this checkout | the overlay packages and the loader |
| `VERIFY_CAP` (`INKLING_VERIFY_CAP`) | `cost:1=0.30,2=0.25,4=0.15,8=0.0` | per batch size, the draft-survival threshold below which drafts are not verified; `0` = verify all 8 rows (used from 5 concurrent requests up, where the expert union is saturated); `off` = stock |
| `DRAFT_BYTES` (`INKLING_DRAFT_BYTES`) | `gemv,fold8,head8,draftmx8` | draft logits on the GEMV, FP8 Markov fold, FP8 draft LM head, MXFP8 draft layers |
| `KERNELS` (`INKLING_KERNELS`), `MOE_TILES` | `moe_tiles,qkvr_gemv` | GB10 NVFP4 tile table for decode; stacked Q/K/V on one GEMV |
| `PICO12` (`INKLING_PICO12`) | `dense,lmhead` | pico's 12-bit packed GEMV for qkvr / wo and the target LM head at <= 16 rows (+1.9 GB memory for the packed copies; the extension builds in ~30 s on the first start) |
| `PREFILL` (`INKLING_PREFILL`) | `qkvr_split,bias_layout,quant_once,partial_bf16` | prefill: per-source qkvr projection, relative bias written in FA4's layout, MoE input quantised once per token, BF16 MoE combine |

Left at SGLang's defaults (measured, see docs/experiments.md): chunked prefill 8,192 tokens (16,384: +2-3 % at
32K-128K, more memory), up to 48 running requests (CUDA graphs captured to 48), radix prefix cache on, overlap
scheduler on, prefill CUDA graphs on.

### Memory budget

GB10 has one 121.6 GiB pool for the GPU and the operating system, so every GiB of KV cache is taken from the host.
`--mem-fraction-static 0.62` gives SGLang a static pool for the weights, the KV cache and the CUDA graphs and leaves
the rest to the host (numbers from the final run's server log):

| | |
|---|---|
| weights (target + draft) | 19.9 GB + 1.4 GB (stock 18.2 + 1.3): the overlays add pico12's packed copies (1.9 GB; the BF16 originals stay for prefill) and FP8 / MXFP8 draft copies |
| KV cache, global layers (and the draft's global layer) | 793,728 tokens, BF16 (24.2 GB + 3.0 GB) |
| KV cache, sliding-window layers | 79,360 tokens (12.1 GB + 0.6 GB): each running or cached request needs one 1,024-token window per layer |
| convolution state (SGLang calls it the Mamba cache) | 715 slots (1.7 GB + 0.9 GB) |
| CUDA graphs | ~7 GB: prefill graphs up to 8,192 tokens (6.2 GB) and decode / verify graphs for 1-48 requests |
| MemAvailable after start / lowest during the full benchmark run | 39.8-40.1 GiB / 39.1-39.3 GiB |

794K tokens of global KV hold three requests at the full 262,144-token context, or 24 agents with 32K each.
`OVERLAY=0` gives back 2.7 GB (843K tokens). Raise `MEM_FRACTION` for more (each 0.01 is ~1.2 GiB, about 28K
more global-KV tokens), but keep MemAvailable above ~6 GiB after warm-up: an out-of-memory on GB10 can hang the box.

### What differs from the published recipes, and why

| | SGLang cookbook (DGX Spark, DSpark) | vLLM recipe (DGX Spark) | here |
|---|---|---|---|
| engine | SGLang | vLLM | **SGLang**: the stock engines decode at the same speed on one Spark (c1 within 3 %, c16 / c24 within run noise); SGLang prefills 1.3-3.4x faster, keeps long prompts in its prefix cache more reliably and leaves twice the host memory ([docs/experiments.md](docs/experiments.md)) |
| memory | `--mem-fraction-static 0.45` (518K KV tokens) | `--gpu-memory-utilization` unset (vLLM's own suggestion would leave the host 0-2 GiB) | `0.62`: 794K KV tokens with the overlays, ~40 GiB MemAvailable after start |
| context | not set (model default) | 262,144 | 262,144 (the model supports 1M; see "Longer context") |
| multimodal | `--enable-multimodal` | on | off (text only; add `--enable-multimodal` for images) |
| images, weights | pulled by tag, weights from the Hub at start | same | pinned image digest, weights at a pinned revision, sha256-checked, offline at start |
| bind address | 0.0.0.0 | 0.0.0.0 | 127.0.0.1 |
| speed overlays | - | - | five packages, on by default (`overlay/`); `OVERLAY=0` = the cookbook engine |

### Longer context, optional FP8 KV

The model supports 1,048,576 tokens; `CTX` sets the per-request limit. The KV pool is shared, not reserved per
request: the default 794K global-layer tokens cover any prompt up to that size. For a full 1M-token request raise
`MEM_FRACTION` (each 0.01 is about 28K tokens; ~0.72 for 1M plus a few short requests) and watch MemAvailable stay
above ~6 GiB; that setting was not measured here. A cold prompt near 1M tokens takes tens of minutes to prefill.
More concurrent long requests can also use an FP8 KV cache. FP8 KV changes the model's numerics (lossy) and is
not part of this recipe's measured profile; Tech2Wild's vLLM recipe measured an FP8 tier (3.3M tokens, needle PASS at
~950K, its tool-calling eval 87.0 vs 88.4 with BF16 KV) and documents the memory budget for it.

## Reproducing the numbers

```bash
git clone https://github.com/knapcio/speedbench speedbench && git -C speedbench checkout 6733653
scripts/launch.sh up                       # or OVERLAY=0 for the stock engine
bench/full.sh MYRUN                        # ~40 min: every table above -> results/MYRUN/
python3 bench/report.py results/MYRUN      # README-style tables
bench/screen.sh MYSCREEN                   # ~10 min screen: speedbench quick, thinking c8, 32K prefill
```

The overlays' own checks run inside the image with the server stopped: `overlay/tools/gpu_check_qkvr.py`,
`gpu_check_pico12.py`, `gpu_check_prefill.py`, `gpu_check.py` (draft package) and the CPU unit tests in
`overlay/tests/` (see `overlay/README.md`).

## Licences

Repository-authored scripts, overlays, tests and documentation are MIT (see `LICENSE`). `overlay/inkling_pico12/
third_party/pico/` holds the GEMV sources by Andy Shu (dsgb10 / pico), unmodified, under their MIT licence (`LICENSE` next to
them); see `NOTICE`. The serving images (SGLang, vLLM) are Apache-2.0 and are used as published, pinned by digest.
The Inkling-2-Mini weights are Apache-2.0 from Thinking Machines Lab.

See [CREDITS.md](CREDITS.md) for the work this builds on.
