"""Packed weights and the small-M calls that go to pico's GEMV.

A Pack holds one weight coded by pico's bf16x_encode (exact: every BF16 value is kept bit for bit; tiles of
8 rows x 32 values store sign + mantissa bytes and 4-bit exponent offsets, ~0.76 of the BF16 bytes on
Inkling's weights). The BF16 original stays where it is: prefill, M > 16 and every other reader use it.

pico's kernel computes out[m, n] = sum_k x[m, k] W[n, k] with BF16 tensor-core MMAs, FP32 accumulation,
split-K over `wk` warps per CTA and a fixed-order reduction; out is FP32 or rounded once to BF16. A given
(row, column) has the same bits whatever M (1..16), the CTA row tiling (mt) or the other rows are: the stacked
qkvr path below relies on that. Against SGLang's stock GEMVs (Triton sm12x_bf16_gemv, b12x, cuBLAS) only the
FP32 summation order differs.

Every function here that runs per forward allocates with torch.empty (the graph pool under capture), launches
on the current stream and never syncs, so it can be captured into a CUDA graph.
"""

import torch

MAX_ROWS = 16  # pico's dsk::kMaxRows: rows of one GEMV call


class Settings:
    max_m = 16  # rows per plain call taken by pico (<= MAX_ROWS)
    stacked_max_m = 16  # rows per source of a stacked qkvr call taken by pico (<= MAX_ROWS)
    wk = 16  # warps along K per CTA (pico's GemvCfg.wk; the probe's setting)
    pdl = True  # launch with programmatic stream serialization (as the probe did)
    require_cuda = True  # CPU unit tests turn this off and install a stand-in EXT


SET = Settings()
EXT = None  # the built extension (build.load), or a stand-in in tests
_batch_invariant = None  # sglang's is_batch_invariant_mode_enabled, set by the patches
_warned = set()
LOG = None  # callable(str), set by the package


class Pico12Fatal(BaseException):
    """A broken patch target or an impossible state: not caught by `except Exception`."""


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        if LOG is not None:
            LOG(msg)


def choose_mt(n):
    """n-tiles of 8 rows per warp: 2 when N allows (wo, head; the probe's choice), else 1 (qkvr's 8600)."""
    return 2 if n % 16 == 0 else 1


def shape_reason(n, k, wk=None):
    """None if pico takes an [n, k] BF16 weight, else why not."""
    wk = SET.wk if wk is None else wk
    if n % 8:
        return f"N={n} not a multiple of 8"
    if k % 32:
        return f"K={k} not a multiple of 32"
    if k // 32 < wk:
        return f"K={k} has fewer than wk={wk} steps of 32"
    if n >= 2**31 or (n // 8) * (k // 32) >= 2**31:
        return f"[{n}, {k}] too large"
    return None


class Pack:
    __slots__ = ("buf", "n", "k", "mt", "raw_tiles", "tiles", "wptr", "label")

    def __init__(self, buf, n, k, raw_tiles, tiles, wptr, label):
        self.buf, self.n, self.k = buf, n, k
        self.mt = choose_mt(n)
        self.raw_tiles, self.tiles, self.wptr, self.label = raw_tiles, tiles, wptr, label

    @property
    def nbytes(self):
        return self.buf.numel()

    @property
    def raw_bytes(self):
        return self.n * self.k * 2

    def __repr__(self):
        return (f"Pack({self.label} [{self.n}x{self.k}] {self.nbytes / 1e6:.2f} MB "
                f"raw_tiles={self.raw_tiles}/{self.tiles} mt={self.mt})")


def encode(weight, label, old=None):
    """Code a [N, K] BF16 weight. With `old` (a reload of the same weight), the new coding is written into
    old's buffer so that CUDA graphs captured on it stay valid; if it no longer fits there is no safe way on,
    so this raises Pico12Fatal."""
    w = weight.detach()
    n, k = w.shape
    why = shape_reason(n, k)
    if why is not None:
        raise ValueError(f"inkling_pico12: {label}: {why}")
    if not w.is_contiguous():
        w = w.contiguous()
    buf, raw, tiles = EXT.encode(w)
    if old is None:
        return Pack(buf, n, k, int(raw), int(tiles), weight.data_ptr(), label)
    if (old.n, old.k) != (n, k) or buf.numel() > old.buf.numel():
        raise Pico12Fatal(
            f"inkling_pico12: reloaded {label} [{n}x{k}] codes to {buf.numel()} bytes, its buffer (captured in "
            f"CUDA graphs) holds {old.buf.numel()}: restart the server"
        )
    old.buf[: buf.numel()].copy_(buf)
    old.raw_tiles, old.wptr = int(raw), weight.data_ptr()
    return old


def mode_ok():
    if torch.compiler.is_compiling():
        return False
    return not (_batch_invariant is not None and _batch_invariant())


def rows_of(x):
    return x.numel() // x.shape[-1] if x.dim() >= 1 and x.shape[-1] else 0


def x_ok(x, k):
    return (
        x.dtype == torch.bfloat16
        and x.dim() >= 1
        and x.shape[-1] == k
        and (x.is_cuda or not SET.require_cuda)
        and not x.requires_grad
    )


def weight_current(weight, pack):
    """The pack was made from this very weight storage (a replaced weight falls back, with a warning)."""
    if weight.data_ptr() == pack.wptr:
        return True
    _warn_once(("wptr", pack.label), f"{pack.label}: weight storage changed since coding; stock path from now on")
    return False


def eligible(x, pack, max_m=None):
    m = rows_of(x)
    return 0 < m <= (SET.max_m if max_m is None else max_m) and x_ok(x, pack.k) and mode_ok()


def as_rows(x, k):
    """x[..., K] as [M, K] with contiguous, 16-byte aligned rows (a copy only when x is not)."""
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != k):
        x2 = x2.contiguous()
    if x2.data_ptr() % 16:
        x2 = x2.clone()
    return x2


def out_ok(out, m, pack, dtype, device):
    return (
        out.dtype == dtype
        and out.device == device
        and out.dim() == 2
        and tuple(out.shape) == (m, pack.n)
        and out.stride(-1) == 1
        and (m == 1 or out.stride(0) == pack.n)
    )


def gemv(x2, pack, out):
    """out[M, N] = x2[M, K] @ W^T, M <= 16, rows of x2 and out contiguous."""
    EXT.gemv(x2, pack.buf, pack.n, pack.k, out, pack.mt, SET.wk, SET.pdl)
    return out


def linear(x, pack, out_dtype, out=None):
    """x[..., K] @ W^T in out_dtype (BF16 or FP32), into `out` ([M, N], checked by the caller) when given."""
    lead = x.shape[:-1]
    x2 = as_rows(x, pack.k)
    if out is None:
        out = torch.empty((x2.shape[0], pack.n), dtype=out_dtype, device=x.device)
    gemv(x2, pack, out)
    return out.view(*lead, pack.n)


# ---- stacked qkvr ---------------------------------------------------------------------------------------


class QkvrPlan:
    """Inkling2Attention.qkvr packed for pico: `full` (all N rows) and, for stacked calls of more than 16 rows,
    `kv` (the k and v rows [kv_lo, kv_hi), adjacent in qkvr's split order q, k, v, r, gate, sinks)."""

    __slots__ = ("full", "kv", "kv_lo", "kv_hi")

    def __init__(self, full, kv, kv_lo, kv_hi):
        self.full, self.kv, self.kv_lo, self.kv_hi = full, kv, kv_lo, kv_hi


def kv_bounds(split_sizes):
    """Columns [lo, hi) of the k and v outputs of the qkvr projection, or None if pico cannot pack them apart
    (bounds not multiples of 8)."""
    if len(split_sizes) < 3:
        return None
    lo = int(split_sizes[0])
    hi = lo + int(split_sizes[1]) + int(split_sizes[2])
    if lo % 8 or hi % 8 or hi <= lo:
        return None
    return lo, hi


def stacked_needs_kv(stacked_max_m):
    """A kv pack is only read when a stacked call has more than 16 rows (3M > 16)."""
    return 3 * stacked_max_m > MAX_ROWS


def stacked_eligible(sources, plan, m_max=None):
    q, k, v = sources
    if q.dim() != 2 or not (q.shape == k.shape == v.shape):
        return False
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16):
        return False
    m = q.shape[0]
    if not 0 < m <= (SET.stacked_max_m if m_max is None else m_max):
        return False
    if 3 * m > MAX_ROWS and plan.kv is None:
        return False
    return x_ok(q, plan.full.k) and mode_ok()


def stacked_project(stacked, m, plan):
    """BF16 [3M, N]: what qkvr(stacked)[0] gives for the three stacked [M, K] sources q, k, v, on every column
    Inkling2Attention._project_sources keeps (q rows: all columns; k rows: the k columns; v rows: the v columns).

    3M <= 16: one call over all 3M rows and all N columns (every element written, as the stock projection).
    3M > 16: the q rows over all N columns, then the k and v rows (2M, in calls of <= 16 rows) over the kv
    pack only, copied into out[M:3M, kv_lo:kv_hi]. Elements outside those (k rows on q/r/gate/sink columns, v
    rows likewise) are left unwritten; _project_sources never returns them. Each written element has the bits
    of the plain per-source call (pico's per-element sums do not depend on M, mt or the other rows)."""
    full = plan.full
    x = as_rows(stacked, full.k)
    out = torch.empty((3 * m, full.n), dtype=torch.bfloat16, device=stacked.device)
    if 3 * m <= MAX_ROWS:
        return gemv(x, full, out)
    gemv(x[:m], full, out[:m])
    kv = plan.kv
    xkv = x[m:]
    tmp = torch.empty((2 * m, kv.n), dtype=torch.bfloat16, device=stacked.device)
    for r0 in range(0, 2 * m, MAX_ROWS):
        r1 = min(2 * m, r0 + MAX_ROWS)
        gemv(xkv[r0:r1], kv, tmp[r0:r1])
    out[m:, plan.kv_lo : plan.kv_hi].copy_(tmp)
    return out


def split_like_stock(out, split_sizes):
    """The (q, k, v, r, gate, sinks) views _project_sources returns for a stacked [3M, N] projection."""
    q_proj, k_proj, v_proj = (chunk.split(split_sizes, dim=-1)[:6] for chunk in out.chunk(3))
    q, _, _, r, gate, sinks = q_proj
    return q, k_proj[1], v_proj[2], r, gate, sinks
