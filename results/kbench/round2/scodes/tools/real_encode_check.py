"""CPU: encode the real NVFP4 expert scales of layers 0-19 exactly as the overlay does, check the exact round trip,
and count escape rows / escape words (SFA words the kernel takes from the stock scales)."""
import json
import sys

sys.path.insert(0, "/work/overlay")
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from inkling_scodes import encode  # noqa: E402

torch.set_num_threads(2)
tot = dict(raw=0, coded=0, esc_rows=0, rows=0, esc_words=0, words=0, weight=0)
ok = True
with safe_open("/model/model-00001-of-00001.safetensors", framework="pt", device="cpu") as h:
    for L in range(20):
        for g in ("w13", "w2"):
            sc = h.get_tensor(f"model.llm.layers.{L}.mlp.experts.{g}_weight_scale").view(torch.uint8)
            wb = h.get_slice(f"model.llm.layers.{L}.mlp.experts.{g}_weight").get_shape()
            e, n, sfk = sc.shape
            k = sfk * 16
            sw = sc.reshape(e, n // 128, 4, 32, sfk // 4, 4).permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, -1)
            codes, bases, stride = encode.encode(torch, sw, n, k)
            back = encode.decode(torch, codes, bases, n, k, stock=sw)
            same = torch.equal(back, sw)
            ok &= same
            # escape words: escape row and some code == 31 (recompute like the kernel)
            mt, kt = n // 128, k // 128
            ch = codes[: e * stride].reshape(e, mt, kt, 5, 128).to(torch.int64)
            nib = torch.stack([ch & 0xF, ch >> 4], dim=-1).reshape(e, mt, kt, 5, 256)
            ones = nib[:, :, :, 0] & nib[:, :, :, 1] & nib[:, :, :, 2] & nib[:, :, :, 3] & nib[:, :, :, 4]
            row = torch.arange(256) % 128
            row = (row & 3) * 32 + (row >> 2)
            esc = (bases.to(torch.int64).reshape(e, mt, 128)[:, :, row] & 0x80) != 0
            ew = int((esc[:, :, None, :] & (ones != 0)).sum())
            er = int((bases >= 0x80).sum())
            st = dict(layer=L, w=g, roundtrip=same, escape_rows=er, escape_words=ew, words=e * mt * kt * 256,
                      blocks_with_escape=int(esc.any(dim=2).sum()), blocks=e * mt)
            print(json.dumps(st), flush=True)
            tot["raw"] += sc.numel(); tot["coded"] += codes.numel() - 640 + bases.numel()
            tot["esc_rows"] += er; tot["rows"] += e * n; tot["esc_words"] += ew; tot["words"] += e * mt * kt * 256
            tot["weight"] += e * n * k // 2
nv = tot["weight"] + tot["raw"]
tot["saving_frac_nvfp4_expert_bytes"] = (tot["raw"] - tot["coded"]) / nv
tot["roundtrip_all"] = bool(ok)
print("TOTAL", json.dumps(tot))
