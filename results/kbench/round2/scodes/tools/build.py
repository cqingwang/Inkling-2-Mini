"""Build (or load from cache) the inkling_scodes JIT module. CPU-only works with FLASHINFER_CUDA_ARCH_LIST=12.1a."""
import sys
import time

sys.path.insert(0, "/work/overlay")
from inkling_scodes import kernel  # noqa: E402

t0 = time.time()
m = kernel.module()
print("built", kernel.source_hash(), "in %.0f s" % (time.time() - t0), m, flush=True)
