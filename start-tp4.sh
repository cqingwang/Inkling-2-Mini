#!/usr/bin/env bash
# Inkling-2-Mini stock SGLang + DSpark on four DGX Sparks.
#
# 与 deepseek 并行运行时只允许创建 inkling-mini-r0..3，不停止或删除任何
# 其他 target。物理环、HCA、GID、rank 与 NCCL_RING_DEV_MAP 全部来自父仓
# config.yaml.common.ring 的受管环境。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.inkling-mini}"
ACTION="${1:-serve}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)
READY_TIMEOUT_SECONDS="${INKLING_READY_TIMEOUT_SECONDS:-3600}"
DIST_READY_TIMEOUT_SECONDS="${INKLING_DIST_READY_TIMEOUT_SECONDS:-120}"
MEMGUARD_GIB="${INKLING_MEMGUARD_GIB:-3.5}"
MEMGUARD_SECONDS="${INKLING_MEMGUARD_SECONDS:-31536000}"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[+] $*"; }
warn() { echo "[!] $*" >&2; }

case "$ACTION" in
  serve|plan) ;;
  *) die "start-tp4.sh 只接受 serve 或 plan" ;;
esac
[ -f "$ENV_FILE" ] || die "缺少受管环境文件: $ENV_FILE"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

container_name_for_rank() { printf '%s-r%s' "$TARGET_CONTAINER_PREFIX" "$1"; }
require() {
  local key="$1"
  [ -n "${!key:-}" ] || die "缺少受管参数 $key"
}

for key in TARGET_ENGINE_ID TARGET_CONTAINER_PREFIX RUNTIME_ASSETS CACHE_DIR \
  MODEL_DIR MODEL_CONTAINER_PATH MODEL_REVISION MODEL_WEIGHT_SHA256 \
  MODEL_DRAFT_SHA256 MODEL_TOKENIZER_SHA256 IMAGE IMAGE_DIGEST IMAGE_PLATFORM \
  IMAGE_ID \
  HEAD_NODE HEAD_MANAGEMENT_IP HEAD_SSH_TARGET WORKER_NODES \
  WORKER_MANAGEMENT_IPS WORKER_SSH_TARGETS WORKER_RANKS RUNTIME_RANK_ORDER \
  ALL_TARGET_CONTAINER_PREFIXES API_HOST PORT API_KEY DIST_PORT NCCL_PORT \
  MIN_MEM_AVAILABLE_GIB CONTAINER_MEMORY NCCL_HOST_DIR NCCL_SHA256 \
  NCCL_PIP_SO NCCL_CONTAINER_DIR NCCL_SO_NAME SERVED_MODEL_NAME \
  MAX_MAMBA_CACHE_SIZE SPECULATIVE_ENABLED DISABLE_CUDA_GRAPH \
  DISABLE_PREFILL_CUDA_GRAPH DISABLE_DECODE_CUDA_GRAPH \
  ENABLE_SINGLE_BATCH_OVERLAP ENABLE_ATTN_TP_INPUT_SCATTERED \
  ENABLE_FUSED_MOE_SUM_ALL_REDUCE SWA_FULL_TOKENS_RATIO CONTAINER_ENV_JSON; do
  require "$key"
done

[ "$TARGET_ENGINE_ID" = "inkling-mini" ] \
  || die "TARGET_ENGINE_ID 必须为 inkling-mini: $TARGET_ENGINE_ID"
[ "$TARGET_CONTAINER_PREFIX" = "inkling-mini" ] \
  || die "TARGET_CONTAINER_PREFIX 必须为 inkling-mini: $TARGET_CONTAINER_PREFIX"
[ "$RUNTIME_ASSETS" = "/opt/runtime_assets/inkling-mini" ] \
  || die "runtime assets 必须为 /opt/runtime_assets/inkling-mini"
[ "$IMAGE_PLATFORM" = "linux/arm64" ] || die "镜像平台必须为 linux/arm64"
[ "$PORT" = "8889" ] || die "inkling-mini API 端口必须为 8889: $PORT"
[ "$DIST_PORT" != "25000" ] || die "inkling-mini dist 端口不得占用 DeepSeek 25000"
[ "$NCCL_PORT" != "25000" ] || die "inkling-mini NCCL 端口不得占用 DeepSeek 25000"
[ "$PORT" != "$DIST_PORT" ] && [ "$PORT" != "$NCCL_PORT" ] \
  && [ "$DIST_PORT" != "$NCCL_PORT" ] \
  || die "inkling-mini API/dist/NCCL 端口不能重复"

read -r -a WORKER_TARGETS <<< "$WORKER_SSH_TARGETS"
read -r -a WORKER_RANK_VALUES <<< "$WORKER_RANKS"
read -r -a WORKER_IP_VALUES <<< "$WORKER_MANAGEMENT_IPS"
read -r -a TARGET_PREFIXES <<< "$ALL_TARGET_CONTAINER_PREFIXES"
[ "${#WORKER_TARGETS[@]}" -eq 3 ] || die "必须提供三个 worker SSH 目标"
[ "${#WORKER_RANK_VALUES[@]}" -eq 3 ] || die "必须提供三个 worker rank"
[ "${#WORKER_IP_VALUES[@]}" -eq 3 ] || die "必须提供三个 worker 管理 IP"
[[ " ${ALL_TARGET_CONTAINER_PREFIXES} " == *" deepseek "* ]] \
  || die "并行启动缺少 deepseek 容器保护清单"

rank_ssh_target() {
  local rank="$1" index
  if [ "$rank" = "0" ]; then
    printf '%s' "$HEAD_SSH_TARGET"
    return
  fi
  for index in "${!WORKER_RANK_VALUES[@]}"; do
    if [ "${WORKER_RANK_VALUES[$index]}" = "$rank" ]; then
      printf '%s' "${WORKER_TARGETS[$index]}"
      return
    fi
  done
  die "未知 rank: $rank"
}

run_rank_shell() {
  local rank="$1" command="$2" target
  target="$(rank_ssh_target "$rank")"
  if [ "$target" = "$HEAD_SSH_TARGET" ]; then
    bash -c "$command"
  else
    ssh "${SSH_OPTS[@]}" "$target" "bash -s" <<< "$command"
  fi
}

container_env_rows() {
  python3 - "$CONTAINER_ENV_JSON" <<'PY'
import json
import sys

try:
    values = json.loads(sys.argv[1])
except json.JSONDecodeError as exc:
    raise SystemExit(f"CONTAINER_ENV_JSON 非法: {exc}") from exc
if not isinstance(values, dict):
    raise SystemExit("CONTAINER_ENV_JSON 顶层必须是对象")
for key in sorted(values):
    value = values[key]
    if not isinstance(value, str):
        raise SystemExit(f"{key} 必须是字符串")
    if any(character in value for character in ("\n", "\t")):
        raise SystemExit(f"{key} 含非法字符")
    print(f"{key}\t{value}")
PY
}

docker_command_for_rank() {
  local rank="$1" key value
  local -a command=(
    docker run -d --name "$(container_name_for_rank "$rank")"
    --network host
    --ipc host
    --gpus all
    --restart no
    --shm-size 16g
    --memory "$CONTAINER_MEMORY"
    --memory-swap "$CONTAINER_MEMORY"
    --ulimit memlock=-1:-1
    --cap-add IPC_LOCK
    --security-opt seccomp=unconfined
    --security-opt label=disable
    --device=/dev/infiniband:/dev/infiniband:rwm
  )
  while IFS=$'\t' read -r key value; do
    [ -n "$key" ] || continue
    command+=("--env=$key=$value")
  done < <(container_env_rows)
  command+=(
    -v "$MODEL_DIR:$MODEL_CONTAINER_PATH:ro"
    -v "$CACHE_DIR:/root/.cache:rw"
    -v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"
    -v "$NCCL_HOST_DIR/$NCCL_SO_NAME:$NCCL_PIP_SO:ro"
    -v "$ROOT/inkling_loadfix.py:/opt/inkling-loadfix/inkling_loadfix.py:ro"
    -v "$ROOT/loadfix/sitecustomize.py:/opt/inkling-loadfix/sitecustomize.py:ro"
    -v "$ROOT/overlay:/opt/inkling-overlay:ro"
    -e PYTHONPATH=/opt/inkling-loadfix:/opt/inkling-overlay
    -e INKLING_LOADFIX=1
    -e INKLING_OVERLAY_SITECUSTOMIZE=/opt/inkling-overlay/sitecustomize.py
    -e INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16
    -e INKLING_ATTN=prefill_swa,prefill_global
    -e INKLING_PFUSE=all
    --entrypoint python3
    "$IMAGE"
    -m inkling_loadfix
    --model-path "$MODEL_CONTAINER_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --tp "$TP_SIZE"
    --ep "$EP_SIZE"
    --nnodes "$NNODES"
    --node-rank "$rank"
    --dist-init-addr "$DIST_INIT_ADDR"
    --nccl-port "$NCCL_PORT"
    --dist-timeout 3600
    --attention-backend fa4
    --dtype bfloat16
    --page-size 128
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --context-length "$CONTEXT_LENGTH"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --grammar-backend xgrammar
    --reasoning-parser inkling
    --tool-call-parser inkling2
    --enable-metrics
    --host "$API_HOST"
    --port "$PORT"
  )
  if [ "$SPECULATIVE_ENABLED" = "true" ]; then
    command+=(--speculative-algorithm DSPARK --speculative-num-draft-tokens 8)
  fi
  if [ "$DISABLE_CUDA_GRAPH" = "true" ] || [ "$DISABLE_DECODE_CUDA_GRAPH" = "true" ]; then
    command+=(--disable-decode-cuda-graph)
  fi
  if [ "$DISABLE_CUDA_GRAPH" = "true" ] || [ "$DISABLE_PREFILL_CUDA_GRAPH" = "true" ]; then
    command+=(--disable-prefill-cuda-graph)
  fi
  if [ "$ENABLE_SINGLE_BATCH_OVERLAP" = "true" ]; then
    command+=(--enable-single-batch-overlap)
  fi
  if [ "$ENABLE_ATTN_TP_INPUT_SCATTERED" = "true" ]; then
    command+=(--enable-attn-tp-input-scattered)
  fi
  if [ "$ENABLE_FUSED_MOE_SUM_ALL_REDUCE" = "true" ]; then
    command+=(--enable-fused-moe-sum-all-reduce)
  fi
  if [ -n "$API_KEY" ]; then
    command+=(--api-key "$API_KEY")
  fi
  printf '%q ' "${command[@]}"
}

run_rank_docker() {
  local rank="$1" command target
  command="$(docker_command_for_rank "$rank")"
  target="$(rank_ssh_target "$rank")"
  if [ "$target" = "$HEAD_SSH_TARGET" ]; then
    bash -c "$command" >/dev/null
  else
    ssh "${SSH_OPTS[@]}" "$target" "$command" </dev/null >/dev/null
  fi
}

start_rank_memguard() {
  local rank="$1" name target
  name="$(container_name_for_rank "$rank")"
  target="$(rank_ssh_target "$rank")"
  if [ "$target" = "$HEAD_SSH_TARGET" ]; then
    bash -c "cd $(printf '%q' "$ROOT") && nohup setsid scripts/memguard.sh $(printf '%q' "$name") $(printf '%q' "$MEMGUARD_GIB") $(printf '%q' "$MEMGUARD_SECONDS") </dev/null >$(printf '%q' "$CACHE_DIR/memguard.log") 2>&1 & pid=\$!; echo \$pid >$(printf '%q' "$CACHE_DIR/memguard.pid")" \
      >/dev/null
  else
    # `ssh -f` detaches locally after auth; remote `exec` keeps the PID that
    # memguard.pid records and avoids the non-interactive bash waiting on its
    # background child while the launcher is blocked in `bash -s`.
    ssh -f "${SSH_OPTS[@]}" "$target" \
      "cd $(printf '%q' "$ROOT") && echo \$\$ >$(printf '%q' "$CACHE_DIR/memguard.pid") && exec scripts/memguard.sh $(printf '%q' "$name") $(printf '%q' "$MEMGUARD_GIB") $(printf '%q' "$MEMGUARD_SECONDS") </dev/null >$(printf '%q' "$CACHE_DIR/memguard.log") 2>&1" \
      >/dev/null
  fi
}

stop_rank_memguard() {
  local rank="$1"
  run_rank_shell "$rank" \
    "if [ -f $(printf '%q' "$CACHE_DIR/memguard.pid") ]; then kill \$(cat $(printf '%q' "$CACHE_DIR/memguard.pid")) >/dev/null 2>&1 || true; fi" \
    >/dev/null 2>&1 || true
}

container_state_for_rank() {
  run_rank_shell "$1" \
    "docker inspect --format '{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}' $(printf '%q' "$(container_name_for_rank "$1")")" \
    2>/dev/null || true
}

preserve_failed_start_logs() {
  local stamp dir rank state logs
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  dir="$ROOT/results/private/failed-starts/$stamp-inkling-mini"
  mkdir -p "$dir"
  for rank in 0 1 2 3; do
    state="$(run_rank_shell "$rank" \
      "docker inspect $(printf '%q' "$(container_name_for_rank "$rank")")" 2>&1 || true)"
    logs="$(run_rank_shell "$rank" \
      "docker logs --timestamps --tail 4000 $(printf '%q' "$(container_name_for_rank "$rank")") 2>&1" 2>&1 || true)"
    printf '%s\n' "$state" >"$dir/rank-${rank}-inspect.txt"
    printf '%s\n' "$logs" >"$dir/rank-${rank}.log"
  done
  warn "失败启动日志已保存: $dir"
}

cleanup_failed_launch() {
  local rc=$? rank
  trap - EXIT
  if [ "$rc" -ne 0 ]; then
    preserve_failed_start_logs
    for rank in 0 1 2 3; do
      stop_rank_memguard "$rank"
      run_rank_shell "$rank" \
        "docker rm -f $(printf '%q' "$(container_name_for_rank "$rank")") >/dev/null 2>&1 || true" \
        >/dev/null 2>&1 || true
    done
    warn "已回滚本次 inkling-mini 启动；deepseek 容器未触碰"
  fi
  exit "$rc"
}
if [ "$ACTION" = serve ]; then
  trap cleanup_failed_launch EXIT
fi

remote_mem_available_gib() {
  run_rank_shell "$1" \
    "awk '/MemAvailable/ {printf \"%.0f\", \$2/1048576}' /proc/meminfo"
}

assert_port_free() {
  local rank="$1" port="$2"
  run_rank_shell "$rank" \
    "if ss -H -ltn 'sport = :$(printf '%s' "$port")' | grep -q .; then exit 1; fi" \
    >/dev/null 2>&1 || die "rank $rank 端口 $port 已被占用"
}

assert_no_same_target_container() {
  local rank="$1" names
  names="$(run_rank_shell "$rank" \
    "docker ps -a --format '{{.Names}}' | awk -v n=$(printf '%q' "$(container_name_for_rank "$rank")") '\$0==n {print}'")"
  [ -z "$names" ] || die "rank $rank 已存在目标容器: $names"
}

preflight() {
  local rank target available image_platform file checksum
  for rank in 0 1 2 3; do
    target="$(rank_ssh_target "$rank")"
    available="$(remote_mem_available_gib "$rank")"
    [ "$available" -ge "$MIN_MEM_AVAILABLE_GIB" ] \
      || die "rank $rank MemAvailable=${available}GiB 低于 ${MIN_MEM_AVAILABLE_GIB}GiB"
    assert_no_same_target_container "$rank"
    run_rank_shell "$rank" "mkdir -p $(printf '%q' "$CACHE_DIR")" >/dev/null
    for file in config.json model.safetensors.index.json \
      model-00001-of-00001.safetensors mtp.safetensors tokenizer.json \
      tiktoken/tokenizer.model; do
      run_rank_shell "$rank" \
        "test -f $(printf '%q' "$MODEL_DIR/$file")" >/dev/null 2>&1 \
        || die "rank $rank 缺少模型文件: $MODEL_DIR/$file"
    done
    checksum="$(run_rank_shell "$rank" \
      "sha256sum $(printf '%q' "$MODEL_DIR/model-00001-of-00001.safetensors") | awk '{print \$1}'")"
    [ "$checksum" = "$MODEL_WEIGHT_SHA256" ] || die "rank $rank 主权重摘要不一致"
    checksum="$(run_rank_shell "$rank" \
      "sha256sum $(printf '%q' "$MODEL_DIR/mtp.safetensors") | awk '{print \$1}'")"
    [ "$checksum" = "$MODEL_DRAFT_SHA256" ] || die "rank $rank DSpark draft 摘要不一致"
    checksum="$(run_rank_shell "$rank" \
      "sha256sum $(printf '%q' "$MODEL_DIR/tokenizer.json") | awk '{print \$1}'")"
    [ "$checksum" = "$MODEL_TOKENIZER_SHA256" ] || die "rank $rank tokenizer 摘要不一致"
    checksum="$(run_rank_shell "$rank" \
      "sha256sum $(printf '%q' "$NCCL_HOST_DIR/$NCCL_SO_NAME") | awk '{print \$1}'")"
    [ "$checksum" = "$NCCL_SHA256" ] || die "rank $rank ring-only NCCL 摘要不一致"
    image_platform="$(run_rank_shell "$rank" \
      "docker image inspect --format '{{.Os}}/{{.Architecture}}' $(printf '%q' "$IMAGE")")" \
      || die "rank $rank 缺少镜像: $IMAGE"
    [ "$image_platform" = "$IMAGE_PLATFORM" ] \
      || die "rank $rank 镜像平台为 $image_platform，期望 $IMAGE_PLATFORM"
    image_id="$(run_rank_shell "$rank" \
      "docker image inspect --format '{{.Id}}' $(printf '%q' "$IMAGE")")" \
      || die "rank $rank 无法读取镜像 ID: $IMAGE"
    [ "$image_id" = "$IMAGE_ID" ] \
      || die "rank $rank 镜像 ID 为 $image_id，期望 $IMAGE_ID"
    run_rank_shell "$rank" \
      "docker run --rm --entrypoint test $(printf '%q' "$IMAGE") -e $(printf '%q' "$NCCL_PIP_SO")" \
      >/dev/null 2>&1 \
      || die "rank $rank 镜像内 NCCL 覆盖路径不存在: $NCCL_PIP_SO"
    assert_port_free "$rank" "$NCCL_PORT"
    if [ "$rank" = "0" ]; then
      assert_port_free "$rank" "$PORT"
      assert_port_free "$rank" "$DIST_PORT"
    fi
    info "rank $rank 预检通过（MemAvailable ${available} GiB）"
  done
}

wait_for_api() {
  local deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
  local -a auth_args=()
  local rank state logs
  if [ -n "$API_KEY" ]; then
    auth_args=(-H "Authorization: Bearer $API_KEY")
  fi
  info "等待 head API: http://127.0.0.1:${PORT}/v1/models（上限 ${READY_TIMEOUT_SECONDS}s）"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    for rank in 0 1 2 3; do
      state="$(container_state_for_rank "$rank")"
      case "$state" in
        true\ *) ;;
        *)
          logs="$(run_rank_shell "$rank" \
            "docker logs --timestamps --tail 120 $(printf '%q' "$(container_name_for_rank "$rank")") 2>&1" 2>&1 || true)"
          die "rank $rank 在 API readiness 前停止: ${state:-inspect-failed}; logs: ${logs: -2400}"
          ;;
      esac
    done
    if [ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "${auth_args[@]}" \
      "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)" = 200 ]; then
      info "API 已响应，执行 10s 稳定性复检"
      sleep 10
      for rank in 0 1 2 3; do
        state="$(container_state_for_rank "$rank")"
        case "$state" in
          true\ *) ;;
          *)
            logs="$(run_rank_shell "$rank" \
              "docker logs --timestamps --tail 120 $(printf '%q' "$(container_name_for_rank "$rank")") 2>&1" 2>&1 || true)"
            die "rank $rank 在 API readiness 后失稳: ${state:-inspect-failed}; logs: ${logs: -2400}"
            ;;
        esac
      done
      if [ "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "${auth_args[@]}" \
        "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null || true)" = 200 ]; then
        info "inkling-mini TP4 UP: http://${API_HOST}:${PORT}/v1/models"
        return
      fi
      die "inkling-mini API 在 readiness 后 10s 内失稳"
    fi
    sleep 10
  done
  die "inkling-mini API 在 ${READY_TIMEOUT_SECONDS}s 内未就绪"
}

wait_for_head_dist_init() {
  local deadline=$(( $(date +%s) + DIST_READY_TIMEOUT_SECONDS )) state
  info "等待 head dist-init 端口 :${DIST_PORT}（上限 ${DIST_READY_TIMEOUT_SECONDS}s）"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ss -H -ltn "sport = :${DIST_PORT}" | grep -q .; then
      info "head dist-init 已监听 :${DIST_PORT}"
      return
    fi
    state="$(container_state_for_rank 0)"
    case "$state" in
      true\ *) ;;
      *) die "rank 0 在 dist-init 监听前停止: ${state:-inspect-failed}" ;;
    esac
    sleep 1
  done
  die "head dist-init 未在 ${DIST_READY_TIMEOUT_SECONDS}s 内监听 :${DIST_PORT}"
}

if [ "$ACTION" = plan ]; then
  for rank in 0 1 2 3; do
    printf 'rank=%s ' "$rank"
    docker_command_for_rank "$rank"
    printf '\n'
  done
  exit 0
fi

preflight
if docker ps --format '{{.Names}}' | grep -Eq '^deepseek-r[0-3]$'; then
  info "DeepSeek 仍在运行；本次只创建 inkling-mini-r0..3"
else
  warn "head 未检测到 deepseek-r0..3；仍按独立 target 启动 inkling-mini"
fi

info "启动 head rank 0"
run_rank_docker 0
start_rank_memguard 0
wait_for_head_dist_init
for rank in 1 2 3; do
  info "启动 rank $rank"
  run_rank_docker "$rank"
  start_rank_memguard "$rank"
done
wait_for_api
trap - EXIT
info "启动完成；停止入口只会删除 inkling-mini-r0..3"
