#!/bin/bash
# GPU window 1: kernel bit-identity + microbench, then the integration check.
set -u
cd /work
if [ ! -f /work/REVIEW_OK ]; then echo "no REVIEW_OK marker: kernel delta not reviewed yet, not running GPU tests"; exit 0; fi
mkdir -p results
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader
[ -s /work/codes_stats.json ] || python3 tools/measure_codes.py | tail -3
python3 tools/gpu_scodes.py --layers 0,6,17,19,0x --bs 1,8,24 --nsteps 2 --iters 30 --json results/gpu1.json; rc1=$?
python3 tools/gpu_integration.py; rc2=$?
echo "RC kernel=$rc1 integration=$rc2"
