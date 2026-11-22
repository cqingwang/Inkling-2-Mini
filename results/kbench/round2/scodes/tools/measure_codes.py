"""Per-row scale-code statistics for the NVFP4 routed experts (layers 0-19) of /model.

Format F5 (the one the kernel implements): per weight row one base byte (row min of the E4M3 scale bytes)
plus a 5-bit code per scale (byte - base), valid iff the row's byte range <= 31.
Prints per layer/tensor: rows, max range, rows with range > 31, distinct bytes, and the byte budget.
"""
import json, struct, sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/model/model-00001-of-00001.safetensors"
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base_off = 8 + n
mm = np.memmap(path, dtype=np.uint8, mode="r")

tot = dict(raw_w=0, raw_s=0, f5=0, bad_rows=0, rows=0)
out = []
for L in range(24):
    for w in ("w13", "w2"):
        k = f"model.llm.layers.{L}.mlp.experts.{w}_weight_scale"
        if k not in hdr:
            continue
        meta = hdr[k]
        if meta["dtype"] != "F8_E4M3":
            print("skip", k, meta["dtype"]); continue
        a, b = meta["data_offsets"]
        E, R, S = meta["shape"]
        s = np.asarray(mm[base_off + a: base_off + b]).reshape(E * R, S)
        wk = hdr[f"model.llm.layers.{L}.mlp.experts.{w}_weight"]
        wb = wk["data_offsets"][1] - wk["data_offsets"][0]
        mn = s.min(axis=1).astype(np.int32); mx = s.max(axis=1).astype(np.int32)
        rng = mx - mn
        bad = int((rng > 31).sum())
        sign = int((s & 0x80).any()); nan = int((s == 0x7F).sum())
        hist = np.bincount(rng, minlength=64)[:48]
        f5 = s.size * 5 // 8 + E * R  # 5-bit codes + one base byte per row
        tot["raw_w"] += wb; tot["raw_s"] += s.size; tot["f5"] += f5; tot["bad_rows"] += bad; tot["rows"] += E * R
        r = dict(layer=L, w=w, E=E, R=R, S=S, weight_bytes=wb, scale_bytes=int(s.size), f5_bytes=f5,
                 range_max=int(rng.max()), range_median=float(np.median(rng)), rows_range_gt31=bad,
                 rows_range_le15=float((rng <= 15).mean()), min_byte=int(s.min()), max_byte=int(s.max()),
                 sign_bits=sign, nan_bytes=nan, distinct=int(len(np.unique(s))))
        out.append(r)
        print(json.dumps(r), flush=True)
raw = tot["raw_w"] + tot["raw_s"]
new = tot["raw_w"] + tot["f5"]
summary = dict(total_expert_bytes_nvfp4=raw, scale_bytes=tot["raw_s"], f5_scale_bytes=tot["f5"],
               saving_bytes=raw - new, saving_frac_of_nvfp4_expert_bytes=(raw - new) / raw,
               rows=tot["rows"], rows_range_gt31=tot["bad_rows"])
print("SUMMARY", json.dumps(summary))
json.dump(dict(rows=out, summary=summary), open("/work/codes_stats.json", "w"), indent=1)
