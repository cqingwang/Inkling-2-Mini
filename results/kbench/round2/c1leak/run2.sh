#!/bin/bash
# kernel-dev window: CPU tests of both packages, then the GPU check + microbenchmarks (bench2.py)
cd /work/overlay
echo "== cpu tests pico12"; python3 -m unittest discover -s tests/pico12 2>&1 | tail -3
echo "== cpu tests draftbytes"; python3 -m unittest discover -s tests/draftbytes 2>&1 | tail -3
cd /work
echo "== bench2"; python3 bench2.py "$@"
