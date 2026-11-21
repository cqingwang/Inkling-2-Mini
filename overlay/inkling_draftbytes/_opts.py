"""INKLING_DRAFT_BYTES parsing.

Comma list, any order, default empty (= the overlay does nothing):

  gemv           draft base logits through the image's sm12x Triton GEMV (<= 16 rows)
  foldk          Markov fold through this package's kernel on the BF16 fold matrix
  fold8[:MODE]   Markov fold from an FP8 e4m3 copy (MODE row | gN | mx; default row)
  head8[:MODE]   draft base logits from an FP8 copy of the LM head (default row)
  draftmx8[:MODE] FP8 copies of the three draft layers' linears (default mx)
  hotvocab:N     draft proposes only from the first N ids of INKLING_DRAFT_HOTVOCAB_FILE
  wide8          this package's FP8 / BF16 kernels run one CTA per output tile for all 64-row blocks (mode 2:
                 one W load per K tile per CTA up to 384 rows; mode 1 / beyond 384: one per row block, L2-hit
                 after the first) instead of a CTA per 64-row block (which re-reads W from DRAM per block);
                 same per-element math (gated bit for bit on the GPU). Applies to whatever head8 / fold8 /
                 foldk / hotvocab / draftmx8 / gemv route through them.

MODE: row = one fp32 scale per output row; gN = one fp32 scale per N inputs of a
row (N a power of two >= 16); mx = MXFP8, one power-of-two (e8m0) scale per 32.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

ENV = "INKLING_DRAFT_BYTES"
HOT_FILE_ENV = "INKLING_DRAFT_HOTVOCAB_FILE"


class OptionError(ValueError):
    pass


@dataclass(frozen=True)
class QuantMode:
    kind: str  # "row" | "group" | "mx"
    group: int = 0  # elements per scale for "group" (and 32 for "mx")

    @property
    def label(self) -> str:
        if self.kind == "row":
            return "row"
        if self.kind == "mx":
            return "mx"
        return f"g{self.group}"


def parse_mode(text: Optional[str], default: str) -> QuantMode:
    text = (text or default).strip().lower()
    if text == "row":
        return QuantMode("row")
    if text == "mx":
        return QuantMode("mx", 32)
    m = re.fullmatch(r"g(\d+)", text)
    if m:
        g = int(m.group(1))
        if g < 16 or g & (g - 1):
            raise OptionError(f"group size must be a power of two >= 16, got {g}")
        return QuantMode("group", g)
    raise OptionError(f"unknown quant mode {text!r} (use row, gN or mx)")


@dataclass(frozen=True)
class Options:
    gemv: bool = False
    foldk: bool = False
    fold8: Optional[QuantMode] = None
    head8: Optional[QuantMode] = None
    draftmx8: Optional[QuantMode] = None
    hotvocab: int = 0
    hotvocab_file: Optional[str] = None
    wide8: bool = False
    # wide8 launches (more than one 64-row block), from the GB10 sweep of 2026-11-21 (tools/gpu_check_wide8.py
    # --tile-sweep; every configuration bit-identical to the default kernel): mode 1, BN 64, 8 warps, 3 stages
    wide8_mode: int = 1  # 1: row blocks outer (W tile re-read from L2); 2: K outer, <= 6 blocks (spills; slower)
    wide8_stages: int = 3  # Triton num_stages (0: Triton's default)
    wide8_bn: int = 64
    wide8_warps: int = 8
    raw: str = ""
    # Tunables (env, read once).
    gemv_max_m: int = 16
    w8_max_m: int = 64
    fold_bn: int = 64
    fold_bk: int = 128
    linear_bn: int = 32
    linear_bk: int = 256
    num_warps: int = 4
    extra: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(
            self.gemv
            or self.foldk
            or self.fold8
            or self.head8
            or self.draftmx8
            or self.hotvocab
        )

    @property
    def fold_kernel(self) -> bool:
        return bool(self.fold8 or self.foldk)

    def describe(self) -> str:
        parts = []
        if self.gemv:
            parts.append("gemv")
        if self.foldk and not self.fold8:
            parts.append("foldk")
        if self.fold8:
            parts.append(f"fold8:{self.fold8.label}")
        if self.head8:
            parts.append(f"head8:{self.head8.label}")
        if self.draftmx8:
            parts.append(f"draftmx8:{self.draftmx8.label}")
        if self.hotvocab:
            parts.append(f"hotvocab:{self.hotvocab}")
        if self.wide8:
            parts.append(f"wide8(mode {self.wide8_mode})")
        return ",".join(parts) or "off"


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as e:
        raise OptionError(f"{name} must be an integer, got {value!r}") from e


def parse(text: Optional[str] = None, environ=None) -> Options:
    environ = os.environ if environ is None else environ
    raw = (environ.get(ENV, "") if text is None else text).strip()
    kw: dict = {"raw": raw}
    for item in filter(None, (t.strip() for t in raw.split(","))):
        name, _, arg = item.partition(":")
        name = name.strip().lower()
        arg = arg.strip() or None
        if name in ("off", "none", "0"):
            continue
        if name == "gemv":
            kw["gemv"] = True
        elif name == "foldk":
            kw["foldk"] = True
        elif name == "fold8":
            kw["fold8"] = parse_mode(arg, "row")
        elif name == "head8":
            kw["head8"] = parse_mode(arg, "row")
        elif name == "draftmx8":
            kw["draftmx8"] = parse_mode(arg, "mx")
        elif name == "wide8":
            kw["wide8"] = True
        elif name == "hotvocab":
            if arg is None or not arg.isdigit() or int(arg) <= 0:
                raise OptionError("hotvocab needs a positive size, e.g. hotvocab:32768")
            kw["hotvocab"] = int(arg)
        else:
            raise OptionError(f"unknown {ENV} option {item!r}")
    if kw.get("hotvocab"):
        path = environ.get(HOT_FILE_ENV, "").strip()
        if not path:
            raise OptionError(f"hotvocab needs {HOT_FILE_ENV} (a JSON list of token ids)")
        kw["hotvocab_file"] = path
    kw["gemv_max_m"] = _int_env("INKLING_DRAFT_GEMV_MAX_M", 16)
    kw["w8_max_m"] = _int_env("INKLING_DRAFT_W8_MAX_M", 64)
    kw["fold_bn"] = _int_env("INKLING_DRAFT_FOLD_BN", 64)
    kw["fold_bk"] = _int_env("INKLING_DRAFT_FOLD_BK", 128)
    kw["linear_bn"] = _int_env("INKLING_DRAFT_LINEAR_BN", 32)
    kw["linear_bk"] = _int_env("INKLING_DRAFT_LINEAR_BK", 256)
    kw["num_warps"] = _int_env("INKLING_DRAFT_NUM_WARPS", 4)
    kw["wide8_mode"] = _int_env("INKLING_DRAFT_WIDE8_MODE", 1)
    kw["wide8_stages"] = _int_env("INKLING_DRAFT_WIDE8_STAGES", 3)
    kw["wide8_bn"] = _int_env("INKLING_DRAFT_WIDE8_BN", 64)
    kw["wide8_warps"] = _int_env("INKLING_DRAFT_WIDE8_WARPS", 8)
    if kw["wide8_bn"] not in (16, 32, 64, 128) or kw["wide8_warps"] not in (1, 2, 4, 8, 16):
        raise OptionError("INKLING_DRAFT_WIDE8_BN must be 16..128 (power of two), _WARPS 1..16 (power of two)")
    if kw["wide8_mode"] not in (1, 2):
        raise OptionError(f"INKLING_DRAFT_WIDE8_MODE must be 1 or 2, got {kw['wide8_mode']}")
    if not 0 <= kw["wide8_stages"] <= 6:
        raise OptionError(f"INKLING_DRAFT_WIDE8_STAGES must be in [0, 6], got {kw['wide8_stages']}")
    for name in ("fold_bn", "fold_bk", "linear_bn", "linear_bk"):
        v = kw[name]
        if v < 16 or v & (v - 1):
            raise OptionError(f"{name} must be a power of two >= 16, got {v}")
    return Options(**kw)
