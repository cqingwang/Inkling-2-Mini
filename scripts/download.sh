#!/usr/bin/env bash
# download.sh [DIR]: thinkingmachines/Inkling-2-Mini at the pinned revision into DIR (default ~/models/Inkling-2-Mini),
# with plain curl (public repository, no token), resumable, then sha256 checks of the large files.
# About 20 GB. On a Spark the download fills the page cache: do not start a model server while it runs.
set -euo pipefail
REV=ed40e7da625f594fc9c4990f5f66be24873c22d7
DST=${1:-${MODEL_DIR:-$HOME/models/Inkling-2-Mini}}
BASE=https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/$REV
mkdir -p "$DST/tiktoken"
cd "$DST"
for f in .gitattributes README.md chat_template.jinja config.json generation_config.json hf_quant_config.json \
         model.safetensors.index.json processor_config.json special_tokens_map.json tokenizer.json \
         tokenizer_config.json tiktoken/tokenizer.model mtp.safetensors model-00001-of-00001.safetensors; do
  echo "fetching $f" >&2
  curl -fSL -C - --retry 5 --progress-bar -o "$f" "$BASE/$f"
done
echo "$REV" > .revision
sha256sum -c - <<SUMS
65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1  model-00001-of-00001.safetensors
7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2  mtp.safetensors
7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6  tokenizer.json
SUMS
echo "Inkling-2-Mini $REV in $DST"
