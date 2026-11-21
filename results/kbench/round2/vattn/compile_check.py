#!/usr/bin/env python3
"""CPU-only compile of the verify kernels for sm_121 (no GPU needed): shared memory from Triton's metadata and
registers / spills from ptxas -v. Usage: python3 compile_check.py (inside the image, /work = this dir)."""
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

TARGET = GPUTarget("cuda", 121, 32)


def sig_of(fn, ptr_dtypes, constexprs):
    sig = {}
    for name in fn.arg_names:
        if name in constexprs:
            sig[name] = "constexpr"
        elif name in ptr_dtypes:
            sig[name] = ptr_dtypes[name]
        elif name == "scale":
            sig[name] = "fp32"
        else:
            sig[name] = "i32"
    return sig


def compile_one(fn, ptrs, consts, warps, stages, tag):
    src = ASTSource(fn=fn, signature=sig_of(fn, ptrs, consts), constexprs=consts)
    k = triton.compile(src, target=TARGET, options={"num_warps": warps, "num_stages": stages})
    smem = k.metadata.shared
    ptx = k.asm["ptx"]
    with tempfile.NamedTemporaryFile("w", suffix=".ptx", delete=False) as f:
        f.write(ptx)
    from triton.backends.nvidia.compiler import get_ptxas

    try:
        ptxas = get_ptxas(TARGET.arch).path
    except TypeError:
        ptxas = get_ptxas().path
    out = subprocess.run([ptxas, "-v", "--gpu-name=sm_121a", f.name, "-o", "/dev/null"], capture_output=True,
                         text=True)
    info = " ".join(re.findall(r"Used \d+ registers|\d+ bytes spill stores|\d+ bytes spill loads", out.stderr))
    print(f"{tag:42s} smem={smem:6d}  {info or out.stderr.strip()[:200]}", flush=True)


def main():
    from inkling_vattn.ops import _inkernel_bias_partials
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import _decode_partials

    common = dict(SQ0=3072, SQ1=128, SK0=131072, SK1=1024, SK2=128, SP=64, HEADS_K=8, QLEN=8, GROUP=3, ROWS=32,
                  D=128, PAGE=128, EXTENT=1024)
    for window in (1023, -1):
        for bn, st in ((32, 2), (32, 3), (64, 2), (64, 3)):
            c = dict(common, SB0=24576, SB1=1024, WINDOW=window, BLOCK_N=bn)
            ptrs = {n: "*bf16" for n in ("Q", "K", "V", "Bias")}
            ptrs.update(PageTable="*i32", SeqLens="*i32", OPartial="*fp32", MPartial="*fp32", LPartial="*fp32")
            compile_one(_decode_partials, ptrs, c, 4, st, f"stock w{window} bn{bn} st{st}")
            wb = triton.next_power_of_2(bn + 7)
            c = dict(common, SR0=3072, SR1=128, D_REL=16, WINDOW=window, BLOCK_N=bn, WB=wb)
            ptrs = {n: "*bf16" for n in ("Q", "K", "V", "R", "Proj")}
            ptrs.update(PageTable="*i32", SeqLens="*i32", OPartial="*fp32", MPartial="*fp32", LPartial="*fp32")
            compile_one(_inkernel_bias_partials, ptrs, c, 4, st, f"inkernel w{window} bn{bn} st{st} wb{wb}")


if __name__ == "__main__":
    main()
