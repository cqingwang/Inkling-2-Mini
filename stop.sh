#!/usr/bin/env bash
# Stop only the Inkling-2-Mini TP4 target. DeepSeek and all other targets remain.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.inkling-mini}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=8)

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "[+] $*"; }

[ -f "$ENV_FILE" ] || die "缺少受管 env: $ENV_FILE"
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

[ "${TARGET_ENGINE_ID:-}" = "inkling-mini" ] || die "TARGET_ENGINE_ID 必须为 inkling-mini"
[ "${TARGET_CONTAINER_PREFIX:-}" = "inkling-mini" ] \
  || die "TARGET_CONTAINER_PREFIX 必须为 inkling-mini"
read -r -a WORKER_TARGETS <<< "${WORKER_SSH_TARGETS:?}"
read -r -a WORKER_RANK_VALUES <<< "${WORKER_RANKS:?}"
[ "${#WORKER_TARGETS[@]}" -eq 3 ] || die "必须提供三个 worker SSH 目标"
[ "${#WORKER_RANK_VALUES[@]}" -eq 3 ] || die "必须提供三个 worker rank"

remote_run() {
  local target="$1" command="$2"
  if [ "$target" = "$HEAD_SSH_TARGET" ]; then
    bash -c "$command"
  else
    ssh "${SSH_OPTS[@]}" "$target" "$command" </dev/null
  fi
}

stop_rank() {
  local target="$1" rank="$2"
  remote_run "$target" \
    "if [ -f $(printf '%q' "$CACHE_DIR/memguard.pid") ]; then kill \$(cat $(printf '%q' "$CACHE_DIR/memguard.pid")) >/dev/null 2>&1 || true; fi; docker rm -f $(printf '%q' "$TARGET_CONTAINER_PREFIX-r$rank") >/dev/null 2>&1 || true"
}

assert_no_rank() {
  local target="$1" names
  names="$(remote_run "$target" \
    "docker ps -a --format '{{.Names}}' | awk -v p=$(printf '%q' "$TARGET_CONTAINER_PREFIX-r") 'index(\$0,p)==1 && substr(\$0,length(p)+1) ~ /^[0-9]+\$/ {print}'")"
  [ -z "$names" ] || die "$target 仍有 inkling-mini 容器: $names"
}

info "stop $TARGET_ENGINE_ID ranks 0..3"
stop_rank "$HEAD_SSH_TARGET" 0
for index in "${!WORKER_TARGETS[@]}"; do
  stop_rank "${WORKER_TARGETS[$index]}" "${WORKER_RANK_VALUES[$index]}"
done

sleep 2
assert_no_rank "$HEAD_SSH_TARGET"
for target in "${WORKER_TARGETS[@]}"; do
  assert_no_rank "$target"
done

code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/v1/models" || true)"
[ "$code" = 000 ] || die "head :${PORT}/v1/models=${code}，停止后仍有服务应答"

remaining="$(docker ps --format '{{.Names}}' | grep -E '^deepseek-r[0-3]$' | sort | tr '\n' ' ' || true)"
info "inkling-mini stopped: target containers=0; deepseek 保留: ${remaining:-none}"
