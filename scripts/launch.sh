#!/usr/bin/env bash
# launch.sh up|down|status|logs: Inkling-2-Mini on one DGX Spark (GB10), one container.
#
#   scripts/launch.sh up                 # default profile: SGLang + DSpark (see README)
#   ENGINE=vllm scripts/launch.sh up     # alternative profile: vLLM + DSpark
#   scripts/launch.sh up --some-flag x   # extra engine flags are appended to the command line
#   scripts/launch.sh status | logs | down
#
# Every setting is an environment variable with the measured default (README, "Every flag explained").
# The server binds to 127.0.0.1 by default; reach it from another machine with `ssh -L 8000:127.0.0.1:8000 spark`
# or set HOST=0.0.0.0 on a trusted LAN. Never expose it through the router.
set -euo pipefail

ENGINE=${ENGINE:-sglang}
NAME=${NAME:-inkling-2-mini}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Inkling-2-Mini}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling-2-mini-$ENGINE}   # JIT / Triton / FlashInfer caches: faster restarts
SERVED=${SERVED:-inkling-2-mini}
CTX=${CTX:-262144}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-3.5}      # memory guard: stop the container below this MemAvailable during start-up
GUARD_SECONDS=${GUARD_SECONDS:-1200}   # how long the guard watches after `up`

# Pinned images (multi-arch index digests; Docker picks linux/arm64 on the Spark) and model revision.
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e}
MODEL_REVISION=ed40e7da625f594fc9c4990f5f66be24873c22d7

# SGLang profile
MEM_FRACTION_USER=${MEM_FRACTION:-}
MEM_FRACTION=${MEM_FRACTION:-0.93}
SWA_RATIO_USER=${SWA_RATIO:-}
SWA_RATIO=${SWA_RATIO:-0.05}                                 # sliding-window pool / full pool tokens (image default 0.1; 48 requests need ~72K)
LOAD_NOMMAP=${LOAD_NOMMAP:-1}                               # read the checkpoint in one pass instead of mmap page faults (start-up ~265 -> ~130 s)
# Speed overlays (overlay/, see overlay/README.md). OVERLAY=0 runs the stock image exactly as the cookbook does.
OVERLAY=${OVERLAY:-1}
OVERLAY_DIR=${OVERLAY_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/overlay}
VERIFY_CAP=${VERIFY_CAP:-policy:/opt/inkling-overlay/inkling_vcap/policies/r2-theta004.json,bsmax=16}  # verify-width policy (off = stock)
DRAFT_BYTES=${DRAFT_BYTES:-gemv,fold8,head8,draftmx8,hotvocab:65536,wide8,w8tile,rnnhoist}  # draft-side byte cuts (empty = stock)
HOTVOCAB_FILE=${HOTVOCAB_FILE:-/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json}  # ranked draft vocabulary
KERNELS=${KERNELS:-moe_tiles,qkvr_gemv}                     # target-side kernel choices (empty = stock)
PREFILL=${PREFILL:-qkvr_split,bias_layout,quant_once,partial_bf16}  # exact prefill levers (empty = stock)
PICO12=${PICO12:-dense,lmhead,stack1}                              # pico packed 12-bit BF16 GEMVs (empty = stock)
CERT_HEAD=${CERT_HEAD:-1}                                   # certified greedy LM-head argmax (0 = off; needs PICO12 lmhead)
WIDE=${WIDE:-dense,lmhead}                                  # packed 12-bit GEMM for 17-256-row verify batches (empty = stock)
WIDE_SCOPE=${WIDE_SCOPE:-verify}                            # wide only on decode/verify forwards
VATTN=${VATTN:-1}                                           # verify attention on the split-KV relative-bias kernel at all batch sizes
ATTN=${ATTN:-prefill_swa,prefill_global}                    # fused relative-bias prefill attention (empty = stock FA4 path)
MOE2=${MOE2:-tiles}                                          # GB10 tile tables for the routed-expert GEMMs (9-128 tokens NVFP4, MXFP8 tactic)
VCAP_BV=${VCAP_BV:-1}                                       # block verification under the verify cap (exact; INKLING_VERIFY_CAP_BV=mask)
VCAP_FASTPROBS=${VCAP_FASTPROBS:-1}                         # draft statistics / sampled accept q from one BF16 read (exact)
VCAP_FUSED=${VCAP_FUSED:-1}                                 # sampled accept straight from the logits, >= 3 requests (exact)
NOEXP=${NOEXP:-1}                                           # dead verify rows get no experts under the verify cap (exact for live rows; inactive when the cap or its remap is off)
KV12=${KV12:-1}                                             # global-layer KV also kept as packed 12-bit pages (exact; side pool)
CERT_WIDE=${CERT_WIDE:-1}                                   # certified greedy LM head for 17-192-row verify batches (exact)
KV12S_DEFAULT=${KV12S_DEFAULT:-1}                           # packed global KV store default for the SGLang overlay profile
if [[ -z ${KV12S:-} ]]; then                                # vLLM and OVERLAY=0 never default to KV12S
  if [[ $ENGINE == sglang && $OVERLAY == 1 ]]; then KV12S=$KV12S_DEFAULT; else KV12S=0; fi
fi                                                          # KV12S=1: packed global target KV, raw BF16 draft lane
case $(printf %s "$KV12S" | tr '[:upper:]' '[:lower:]') in   # one spelling for the shell checks and the overlay switch
  1|true|yes|on) KV12S=1 ;;
  0|false|no|off|'') KV12S=0 ;;
  *) echo "KV12S must be 0 or 1 (got '$KV12S')" >&2; exit 2 ;;
esac
case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in                # the store is chosen here, so engine flags and overlay agree
  *INKLING_KV12*)                                            # any INKLING_KV12* name, with or without a value
    echo "set KV12S / KV12S_POOL / KV12 / KV12_SLOTS instead of passing the KV store's variables in SGLANG_ENV" >&2; exit 2 ;;
esac
[[ $KV12S == 1 ]] && case " ${SGLANG_ENV:-} ${SGLANG_ARGS:-} $* " in
  *enable-unified-memory*|*cuda-graph-backend-prefill*)
    echo "with KV12S=1, choose the pool with KV12S_POOL instead of passing its engine flags" >&2; exit 2 ;;
esac
KV12S_POOL=${KV12S_POOL:-static}                           # static: stock (non-unified) pools; unified: --enable-unified-memory (v2.7.1)
case $KV12S_POOL in static|unified) ;; *) echo "KV12S_POOL must be static or unified (got '$KV12S_POOL')" >&2; exit 2 ;; esac
KV12S_OVERFLOW_HEADS=${KV12S_OVERFLOW_HEADS:-2048}          # exact page-head overflow slots per target visit (32 KiB each)
KV12S_FALLBACK_PAGES=${KV12S_FALLBACK_PAGES:-64}            # raw fallback pages per visit if that arena is exhausted (512 KiB each)
KV12_SLOTS=${KV12_SLOTS:-2048}                              # packed side-pool pages per global layer (2048 = 6.1 GiB, one 256K request)
SCODES=${SCODES:-1}                                          # NVFP4 expert scale codes: 5-bit codes + row base, expanded in the GEMM (exact)
PFUSE=${PFUSE:-all}                                         # prefill glue fusion (empty = stock)
MOE_TILES=${MOE_TILES:-1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256}
SPEC=${SPEC:-dspark}                   # dspark | none
SGLANG_ARGS=${SGLANG_ARGS:-}
# vLLM profile
GPU_UTIL=${GPU_UTIL:-0.80}
VLLM_ARGS=${VLLM_ARGS:-}

cmd=${1:-}; shift || true

case $cmd in
  down)
    docker stop "$NAME" >/dev/null 2>&1 || true
    docker rm "$NAME" >/dev/null 2>&1 || true
    pkill -f "memguard.sh $NAME " 2>/dev/null || true
    echo "stopped $NAME"; exit 0 ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format '{{.Names}}  {{.Image}}  {{.Status}}'
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && echo "API: ready on :$PORT" || echo "API: not ready"
    awk '/MemAvailable/ {printf "MemAvailable: %.1f GiB\n", $2/1048576}' /proc/meminfo
    exit 0 ;;
  logs)
    exec docker logs -f "$NAME" ;;
  up) ;;
  *) echo "usage: $0 up|down|status|logs [extra engine flags]" >&2; exit 2 ;;
esac

kv12s_args=(); kv12s_env=()
if [[ $KV12S == 1 ]]; then
  [[ $ENGINE == sglang && $OVERLAY == 1 ]] || {
    echo "KV12S=1 requires ENGINE=sglang and OVERLAY=1" >&2; exit 2; }
  KV12=0
  if [[ $KV12S_POOL == static ]]; then
    # The packed store replaces the global BF16 pool and the KV12 side copy; with the draft's
    # window lanes priced per window token (static.DRAFT_WINDOW_LANES), 0.89 sizes ~2.19M
    # tokens and keeps the kvstress worst case above the 3.5 GiB guard (prefill graphs kept).
    [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.89
    # ~79K window tokens, as v2.7 and v2.7.1 (48 requests' windows plus an 8K chunk).
    [[ -n $SWA_RATIO_USER ]] || SWA_RATIO=0.037
    kv12s_args=(--max-running-requests 48 --cuda-graph-max-bs-decode=48)
  else
    # The unified pool leaves less post-capture headroom than the stock pool at the same
    # fraction; 0.91 keeps MemAvailable above the guard during 8K prefill chunks.
    [[ -n $MEM_FRACTION_USER ]] || MEM_FRACTION=0.91
    # 3.5 % of 2.26M FULL tokens still covers 48 requests' windows (think48 + kvstress measured at 0.035).
    [[ -n $SWA_RATIO_USER ]] || SWA_RATIO=0.035
    kv12s_args=(--enable-unified-memory --cuda-graph-backend-prefill=disabled --max-running-requests 48
                --cuda-graph-max-bs-decode=48)
  fi
  kv12s_env=(-e SGLANG_RAGGED_VERIFY_MODE=static -e INKLING_KV12S_POOL="$KV12S_POOL")
fi

[[ -f $MODEL_DIR/config.json && -f $MODEL_DIR/mtp.safetensors ]] || {
  echo "no model in $MODEL_DIR (run scripts/download.sh first; mtp.safetensors is the DSpark draft)" >&2; exit 1; }
if [[ -f $MODEL_DIR/.revision && $(cat "$MODEL_DIR/.revision") != "$MODEL_REVISION" ]]; then
  echo "warning: $MODEL_DIR is revision $(cat "$MODEL_DIR/.revision"), this recipe was measured on $MODEL_REVISION" >&2
fi
if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "$NAME is already running (scripts/launch.sh down first)" >&2; exit 1
fi
# One model server per Spark: unified memory is shared by the GPU and the OS.
others=$(docker ps --format '{{.Names}} {{.Image}}' | grep -Ei 'sglang|vllm|tensorrt|llama|ollama' || true)
[[ -z $others ]] || { echo "another model server is running, stop it first:"$'\n'"$others" >&2; exit 1; }
docker rm "$NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE_DIR"

common=(docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 16g --restart no
        -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/root/.cache
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1)

if [[ $ENGINE == sglang ]]; then
  nommap_arg=(); nommap_env=()
  # one-pass checkpoint read; large frees go straight back to the OS so the KV pool is sized on the real free memory
  [[ $LOAD_NOMMAP == 1 ]] && { nommap_arg=(--weight-loader-disable-mmap); nommap_env=(-e MALLOC_MMAP_THRESHOLD_=1048576); }
  spec=()
  [[ $SPEC == dspark ]] && spec=(--speculative-algorithm DSPARK --speculative-num-draft-tokens 8)
  [[ $SPEC == dspark && $OVERLAY == 1 && $VCAP_BV == 1 ]] && spec+=(--speculative-use-block-verification)
  ov=()
  if [[ $OVERLAY == 1 ]]; then
    [[ -f $OVERLAY_DIR/sitecustomize.py ]] || { echo "OVERLAY=1 but no overlay at $OVERLAY_DIR" >&2; exit 1; }
    ov=(-v "$OVERLAY_DIR":/opt/inkling-overlay:ro -e PYTHONPATH=/opt/inkling-overlay
        -e INKLING_VERIFY_CAP="$VERIFY_CAP" -e INKLING_VERIFY_CAP_LOG_EVERY="${VERIFY_CAP_LOG_EVERY:-0}"
        -e INKLING_DRAFT_BYTES="$DRAFT_BYTES" -e INKLING_DRAFT_HOTVOCAB_FILE="$HOTVOCAB_FILE" -e INKLING_KERNELS="$KERNELS" -e INKLING_MOE_TILES="$MOE_TILES"
        -e INKLING_PICO12="$PICO12" -e INKLING_PREFILL="$PREFILL" -e INKLING_PREFILL_QKVR_OUT=strided
        -e INKLING_CERT_HEAD="$CERT_HEAD" -e INKLING_WIDE="$WIDE" -e INKLING_WIDE_SCOPE="$WIDE_SCOPE"
        -e INKLING_VATTN="$VATTN" -e INKLING_ATTN="$ATTN" -e INKLING_PFUSE="$PFUSE" -e INKLING_MOE2="$MOE2" -e INKLING_SCODES="$SCODES"
        -e INKLING_VERIFY_CAP_FASTPROBS="$VCAP_FASTPROBS" -e INKLING_VERIFY_CAP_FUSED_ACCEPT="$VCAP_FUSED" -e INKLING_NOEXP="$NOEXP"
        -e INKLING_KV12="$KV12" -e INKLING_KV12_MAX_PAGES="$KV12_SLOTS"
        -e INKLING_CERT_WIDE="$CERT_WIDE" -e INKLING_KV12S="$KV12S" -e INKLING_KV12S_OVERFLOW_HEADS="$KV12S_OVERFLOW_HEADS"
        -e INKLING_KV12S_FALLBACK_PAGES="$KV12S_FALLBACK_PAGES" ${kv12s_env[@]+"${kv12s_env[@]}"})
  fi
  run=("${common[@]}" ${ov[@]+"${ov[@]}"}
       -e MAX_JOBS=4
       -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1
       -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1
       -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
       -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1
       ${SGLANG_ENV:-}
       ${nommap_env[@]+"${nommap_env[@]}"}
       --entrypoint python3 "$SGLANG_IMAGE"
       -m sglang.launch_server --model-path /model --served-model-name "$SERVED"
       --tp 1 --ep 1 --attention-backend fa4 --dtype bfloat16 --page-size 128
       ${spec[@]+"${spec[@]}"} ${kv12s_args[@]+"${kv12s_args[@]}"}
       --mem-fraction-static "$MEM_FRACTION" --context-length "$CTX"
       --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2
       --enable-metrics --host "$HOST" --port "$PORT" ${nommap_arg[@]+"${nommap_arg[@]}"} --swa-full-tokens-ratio "$SWA_RATIO"
       --max-mamba-cache-size 512 --max-prefill-tokens 8192 $SGLANG_ARGS "$@")
elif [[ $ENGINE == vllm ]]; then
  specj='{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
  spec=(--speculative-config "$specj")
  [[ $SPEC == none ]] && spec=()
  run=("${common[@]}" ${VLLM_ENV:-}
       "$VLLM_IMAGE"
       /model --served-model-name "$SERVED" --tensor-parallel-size 1
       --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL"
       --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling
       ${spec[@]+"${spec[@]}"}
       --host "$HOST" --port "$PORT" $VLLM_ARGS "$@")
else
  echo "ENGINE must be sglang or vllm" >&2; exit 2
fi

echo "+ ${run[*]}" >&2
"${run[@]}" >/dev/null
echo "started $NAME ($ENGINE) on $HOST:$PORT; first start ~6-8 min (weights + JIT + CUDA graphs), later starts faster"

# Memory guard: on GB10 an out-of-memory during start-up can hang the whole box until a power cycle.
# scripts/memguard.sh stops the container if MemAvailable stays below MIN_AVAIL_GIB for two samples (2 s apart).
nohup "$(dirname "${BASH_SOURCE[0]}")/memguard.sh" "$NAME" "$MIN_AVAIL_GIB" "$GUARD_SECONDS" >"$CACHE_DIR/memguard.log" 2>&1 &
echo "memory guard: stops $NAME if MemAvailable < ${MIN_AVAIL_GIB} GiB in the next ${GUARD_SECONDS} s (log $CACHE_DIR/memguard.log)"
