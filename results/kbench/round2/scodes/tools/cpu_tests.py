#!/usr/bin/env python3
"""CPU tests for inkling_scodes (SGLang image, no GPU needed): format round trip, an independent per-thread model
of the kernel's decode, range/sign rejection, and install()'s source checks against the image's sglang."""

import os
import random
import sys

sys.path.insert(0, "/work/overlay")

import torch  # noqa: E402

from inkling_scodes import encode  # noqa: E402

fails = 0


def check(cond, what):
    global fails
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    fails += 0 if cond else 1


def swizzle(scale):  # sglang.srt.layers.quantization.utils.swizzle_blockscale for aligned shapes
    b, m, k = scale.shape
    return scale.reshape(b, m // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(b, m * k)


def random_scales(e, n, sfk, seed, spread=31, outliers=0.002):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, 0x7E - spread, (e, n, 1), generator=g)
    off = torch.randint(0, spread + 1, (e, n, sfk), generator=g)
    out = (base + off).clamp(max=0x7E)
    # a few far outliers (escape rows), like the real tail of small scales
    mask = torch.rand((e, n, sfk), generator=g) < outliers
    far = torch.randint(0, 0x7F, (e, n, sfk), generator=g)
    return torch.where(mask, far, out).to(torch.uint8)


def kernel_thread_model(codes, bases, e, mt, kc, n, k, stock):
    """What consumer thread t writes for chunk (e, mt, kc): plain-int re-implementation of scodes_decode."""
    kt = k // 128
    stride = (n // 128) * kt * 640
    chunk = codes[e * stride + (mt * kt + kc) * 640: e * stride + (mt * kt + kc + 1) * 640].tolist()
    words = []
    for t in range(256):
        row = (t & 3) * 32 + ((t & 127) >> 2)
        braw = int(bases[e, mt * 128 + row])
        v = 0x01010101 * (braw & 0x7F)
        ones = 0xF
        for b in range(5):
            nib = (chunk[b * 128 + (t >> 1)] >> ((t & 1) * 4)) & 0xF
            ones &= nib
            v += ((nib * 0x00204081) & 0x01010101) << b
        if (braw & 0x80) and ones:
            off = (mt * (k // 128) + kc) * 1024 + 4 * t   # stock word t of this chunk
            v = int.from_bytes(bytes(stock[e, off:off + 4].tolist()), "little")
        words.append(v & 0xFFFFFFFF)
    return b"".join(w.to_bytes(4, "little") for w in words)


def main():
    for (e, n, k, seed) in ((2, 256, 256, 1), (3, 384, 1152, 2), (2, 2304, 2304, 3)):
        logical = random_scales(e, n, k // 16, seed)
        sw = swizzle(logical)
        codes, bases, stride = encode.encode(torch, sw, n, k, experts_per_pass=2)
        check(stride == (n // 128) * (k // 128) * 640 and codes.numel() == e * stride + 640,
              f"[{e},{n},{k}] sizes")
        check(torch.equal(encode.decode(torch, codes, bases, n, k, stock=sw), sw),
              f"[{e},{n},{k}] decode(encode) == input ({int((bases >= 0x80).sum())} escape rows)")
        check(torch.equal(encode.logical_from_swizzled(torch, sw, n, k), logical), f"[{e},{n},{k}] unswizzle")
        plain = bases < 0x80
        check(torch.equal(bases[plain], logical.amin(dim=2)[plain]), f"[{e},{n},{k}] plain bases are row minima")
        check(bool((codes[e * stride:] == 0).all()), f"[{e},{n},{k}] zero tail")
        rnd = random.Random(seed)
        for _ in range(6):
            ee, mt, kc = rnd.randrange(e), rnd.randrange(n // 128), rnd.randrange(k // 128)
            got = kernel_thread_model(codes, bases, ee, mt, kc, n, k, sw)
            off = (mt * (k // 128) + kc) * 1024
            want = bytes(sw[ee, off:off + 1024].tolist())
            check(got == want, f"[{e},{n},{k}] thread model chunk e{ee} mt{mt} kc{kc} == swizzled bytes")

    # Range 32 becomes an escape row with the best window; exact round trip; escape thread model.
    logical = random_scales(1, 128, 16, 5, outliers=0)
    logical[0, 7, :] = torch.arange(16, dtype=torch.uint8) + 60
    logical[0, 7, 0] = 20
    sw = swizzle(logical)
    codes, bases, _ = encode.encode(torch, sw, 128, 256)
    b7 = int(bases[0, 7])
    check(b7 & 0x80 and (b7 & 0x7F) <= 60 and (b7 & 0x7F) + 30 >= 75, f"escape row window covers 60..75 ({b7:#x})")
    check(torch.equal(encode.decode(torch, codes, bases, 128, 256, stock=sw), sw), "escape row round trip")
    check(kernel_thread_model(codes, bases, 0, 0, 0, 128, 256, sw) == bytes(sw[0, :1024].tolist()),
          "escape row thread model")
    # sign bit / NaN code must be rejected.
    logical = random_scales(1, 128, 16, 6)
    logical[0, 3, 3] = 0x7F
    try:
        encode.encode(torch, swizzle(logical), 128, 256)
        check(False, "0x7F rejected")
    except encode.ScodesError:
        check(True, "0x7F rejected")
    try:
        encode.encode(torch, swizzle(random_scales(1, 128, 8, 7)), 128, 128 + 0)  # k=128 ok
        check(True, "k=128 accepted")
    except encode.ScodesError as exc:
        check(False, f"k=128 accepted ({exc})")

    # install(): source checks pass against this image, and nothing imports sglang/torch heavy paths early.
    import inkling_scodes

    try:
        inkling_scodes.check_sources()
        check(True, "source checks pass on the image's sglang")
    except RuntimeError as exc:
        check(False, f"source checks pass ({exc})")
    os.environ["INKLING_SCODES"] = ""
    inkling_scodes.install()
    check("sglang.srt.models.inkling_common.inkling2_moe" not in sys.modules, "disabled install imports nothing")

    print("RESULT", "PASS" if fails == 0 else f"FAIL ({fails})", flush=True)
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
