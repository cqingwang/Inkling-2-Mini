#!/bin/bash
# CPU tests: HEAD packages (base/) vs ours, failures listed; then bench2 (draft part unless args say otherwise)
for o in /work/base/overlay /work/overlay; do
  cd $o
  for t in pico12 draftbytes; do
    echo "== cpu tests $t in $o"; python3 -m unittest discover -s tests/$t 2>&1 | grep -E "^(FAIL|ERROR):|^Ran|^OK|^FAILED" | head -20
  done
done
cd /work
echo "== bench2"; python3 bench2.py "$@"
