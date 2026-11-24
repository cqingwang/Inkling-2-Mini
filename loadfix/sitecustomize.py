"""Apply the Inkling weight-loader compatibility shim in every Python process.

SGLang loads weights in a spawned scheduler child, not in the launcher process.
The child imports sitecustomize before the model loader, so this is the only
reliable place to install the mmap-to-anonymous-copy shim for that process.
The container is dedicated to Inkling, so installing unconditionally is safe.
"""

from __future__ import annotations

import os
import runpy
import sys

import inkling_loadfix


inkling_loadfix.install()
sys.stderr.write(f"[inkling-loadfix] installed pid={os.getpid()}\n")

overlay_sitecustomize = os.environ.get("INKLING_OVERLAY_SITECUSTOMIZE", "").strip()
if overlay_sitecustomize:
    sys.stderr.write(
        f"[inkling-loadfix] overlay sitecustomize={overlay_sitecustomize} pid={os.getpid()}\n"
    )
    runpy.run_path(overlay_sitecustomize, run_name="__main__")
