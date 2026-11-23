"""Stream-ordered CUDA writes and side-slot lifetime for packed page envelopes."""

import torch
import triton
import triton.language as tl

from .storage import CAP, CHUNK, DIM, EXP_LO, PACK_BATCH, PAGE, Layer, STAGE_MASK, STAGE_SHIFT, raw_lane_bytes


@triton.jit
def _claim(Locks, page, SLOTS: tl.constexpr):
    # Slot ownership is independent of physical page IDs. A whole-page move can
    # transfer the descriptor without rewriting the slot or its lock.
    slot = page.to(tl.int32) % SLOTS
    old = tl.atomic_cas(Locks + slot, 0, 1, sem="acquire")
    probes = 1
    while (old != 0) & (probes < SLOTS):
        slot = (slot + 1) % SLOTS
        old = tl.atomic_cas(Locks + slot, 0, 1, sem="acquire")
        probes += 1
    return tl.where(old == 0, slot, -1)


@triton.jit
def _direct(Loc, i, loc, n_rows, PAGE: tl.constexpr):
    row = loc % PAGE
    start = i - row
    end = start + PAGE - 1
    lo = tl.load(Loc + start, start >= 0, other=0)
    hi = tl.load(Loc + end, end < n_rows, other=0)
    return (lo == loc - row) & (hi == loc + PAGE - 1 - row)


@triton.jit(debug=True, do_not_specialize=["n_rows"])
def _reserve(Loc, Flags, Locks, n_rows, NPAGES: tl.constexpr,
             PAGE: tl.constexpr, STRIDE: tl.constexpr, STAGES: tl.constexpr,
             SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr, TENTATIVE: tl.constexpr):
    i = tl.program_id(0)
    loc = tl.load(Loc + i).to(tl.int64)
    page, row = loc // PAGE, loc % PAGE
    tl.device_assert((loc <= 0) | ((page > 0) & (page < NPAGES)), "kv12s: invalid physical write row")
    if (page > 0) & (page < NPAGES):
        last = tl.load(Loc + i - 1, i > 0, other=0)
        if (row == 0) | (loc != last + 1):
            ptr = Flags + page * (STRIDE // 4)
            flags = tl.load(ptr)
            tl.device_assert(flags != 1, "kv12s: write to a committed page")
            direct = False
            if not TENTATIVE:
                direct = _direct(Loc, i, loc, n_rows, PAGE)
            if not direct:
                if flags == 0:
                    tl.device_assert(row == 0, "kv12s: staged append is missing its prefix")
                    slot = _claim(Locks, page, STAGES)
                    tl.device_assert(slot >= 0, "kv12s: staging arena exhausted")
                    if slot >= 0:
                        # This provisional descriptor is visible only to the
                        # scatter until the final publication kernel runs.
                        tl.store(ptr, (slot << SHIFT) | 2)
                else:
                    tl.device_assert((row < (flags & COUNT_MASK)) & ((flags >> SHIFT) < STAGES),
                                     "kv12s: staged append is missing its prefix")


@triton.jit(do_not_specialize=["n_rows"])
def _scatter(Loc, K, V, Flags, Stage, n_rows, NPAGES: tl.constexpr,
             PAGE: tl.constexpr, H: tl.constexpr, D: tl.constexpr, STRIDE: tl.constexpr,
             K0: tl.constexpr, K1: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr,
             STAGES: tl.constexpr, SHIFT: tl.constexpr, TENTATIVE: tl.constexpr,
             BLOCK: tl.constexpr):
    i, kind = tl.program_id(0), tl.program_id(1)
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc // PAGE
    if (page > 0) & (page < NPAGES):
        direct = False
        if not TENTATIVE:
            direct = _direct(Loc, i, loc, n_rows, PAGE)
        if not direct:
            flags = tl.load(Flags + page * (STRIDE // 4))
            slot = flags >> SHIFT
            if (flags >= 2) & (slot < STAGES):
                dim = tl.arange(0, BLOCK)
                if kind == 0:
                    words = tl.load(K + i.to(tl.int64) * K0 + dim // D * K1 + dim % D,
                                    dim < H * D, other=0)
                else:
                    words = tl.load(V + i.to(tl.int64) * V0 + dim // D * V1 + dim % D,
                                    dim < H * D, other=0)
                base = ((slot.to(tl.int64) * 2 + kind) * PAGE + loc % PAGE) * H * D
                tl.store(Stage + base + dim, words, dim < H * D)


@triton.jit
def _escape(bits, LO: tl.constexpr):
    exp = (bits >> 7) & 255
    inline = (exp >= LO) & (exp < LO + 15)
    return tl.where(inline, exp - LO, 15), (exp != 0) & ~inline


@triton.jit(debug=True, do_not_specialize=["n_rows"])
def _pack(Loc, K, V, SM, EX, ESC, OFF, Refs, Flags, Stage, Overflow, Locks,
          n_rows, NPAGES: tl.constexpr, PAGE: tl.constexpr, H: tl.constexpr,
          D: tl.constexpr, STRIDE: tl.constexpr, K0: tl.constexpr, K1: tl.constexpr,
          V0: tl.constexpr, V1: tl.constexpr, OVERFLOW: tl.constexpr, SHIFT: tl.constexpr,
          COUNT_MASK: tl.constexpr, CAP: tl.constexpr, CHUNK: tl.constexpr, LO: tl.constexpr,
          PROMOTE: tl.constexpr):
    i, head, kind = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc if PROMOTE else loc // PAGE
    if (page > 0) & (page < NPAGES) & (PROMOTE | (loc % PAGE == PAGE - 1)):
        base = page * STRIDE
        flags = tl.load(Flags + base // 4)
        direct = False
        if not PROMOTE:
            direct = _direct(Loc, i, loc, n_rows, PAGE)
        if (flags >= 2) | direct:
            if not PROMOTE or ((flags & COUNT_MASK) == PAGE + 1):
                pair = tl.arange(0, PAGE * D // 2)
                row = pair // (D // 2)
                dim = pair % (D // 2) * 2
                if direct:
                    src = i.to(tl.int64) - (PAGE - 1) + row
                    if kind == 0:
                        ptr = K + src * K0 + head * K1 + dim
                        a, b = tl.load(ptr), tl.load(ptr + 1)
                    else:
                        ptr = V + src * V0 + head * V1 + dim
                        a, b = tl.load(ptr), tl.load(ptr + 1)
                else:
                    slot = (flags >> SHIFT).to(tl.int64)
                    ptr = Stage + (((slot * 2 + kind) * PAGE + row) * H + head) * D + dim
                    a, b = tl.load(ptr), tl.load(ptr + 1)
                a, b = a.to(tl.uint16).to(tl.int32), b.to(tl.uint16).to(tl.int32)
                ea, xa = _escape(a, LO)
                eb, xb = _escape(b, LO)
                ranks = tl.cumsum(xa.to(tl.int32) + xb.to(tl.int32))
                count = tl.sum(xa.to(tl.int32) + xb.to(tl.int32), 0)
                lane = head * 2 + kind
                if count > CAP:
                    slot = _claim(Locks, page + lane, OVERFLOW)
                    tl.device_assert(slot >= 0, "kv12s: exact page-head overflow arena exhausted")
                    if slot >= 0:
                        dst = Overflow + slot.to(tl.int64) * (PAGE * D) + pair * 2
                        tl.store(dst, a)
                        tl.store(dst + 1, b)
                        tl.store(Refs + base // 4 + lane, slot + 1)
                else:
                    elem = ((kind * PAGE + row) * H + head) * D + dim
                    sign_a = ((a >> 8) & 128) | (a & 127)
                    sign_b = ((b >> 8) & 128) | (b & 127)
                    tl.store(SM + base + elem, sign_a)
                    tl.store(SM + base + elem + 1, sign_b)
                    tl.store(EX + base + elem // 2, ea | (eb << 4))
                    stream = ESC + base // 4 + lane * CAP
                    entry = (row << 23) | (dim << 16)
                    tl.store(stream + ranks - xb.to(tl.int32) - 1, entry | a, xa)
                    tl.store(stream + ranks - 1, entry | (1 << 16) | b, xb)
                    offsets = OFF + base // 4 + lane * (PAGE // CHUNK + 1)
                    tl.store(offsets, 0)
                    tl.store(offsets + row // CHUNK + 1, ranks,
                             (row % CHUNK == CHUNK - 1) & (dim == D - 2))
                    tl.store(Refs + base // 4 + lane, 0)


@triton.jit(do_not_specialize=["n_rows"])
def _publish(Loc, Flags, Locks, n_rows, NPAGES: tl.constexpr, PAGE: tl.constexpr,
             STRIDE: tl.constexpr, SHIFT: tl.constexpr, COUNT_MASK: tl.constexpr,
             TENTATIVE: tl.constexpr, PROMOTE: tl.constexpr):
    i = tl.program_id(0)
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc if PROMOTE else loc // PAGE
    if (page > 0) & (page < NPAGES):
        ptr = Flags + page * (STRIDE // 4)
        flags = tl.load(ptr)
        if PROMOTE:
            end = (flags & COUNT_MASK) - 1
            publish = (flags >= 2) & (end == PAGE)
        else:
            end = loc % PAGE + 1
            following = tl.load(Loc + i + 1, i + 1 < n_rows, other=0)
            publish = (end == PAGE) | (following != loc + 1)
        if publish:
            slot = flags >> SHIFT
            if not TENTATIVE and end == PAGE:
                tl.store(ptr, 1)
                if flags >= 2:
                    tl.atomic_xchg(Locks + slot, 0, sem="release")
            elif flags >= 2:
                tl.store(ptr, (slot << SHIFT) | (end + 1))


@triton.jit(debug=True)
def _trim(Loc, Commit, Flags, Locks, WIDTH: tl.constexpr,
          NPAGES: tl.constexpr, PAGE: tl.constexpr, STRIDE: tl.constexpr, SHIFT: tl.constexpr,
          COUNT_MASK: tl.constexpr):
    i = tl.program_id(0)
    loc = tl.load(Loc + i).to(tl.int64)
    page = loc // PAGE
    col = i % WIDTH
    last_row = (loc % PAGE).to(tl.int32)
    if (page > 0) & (page < NPAGES) & ((last_row == PAGE - 1) | (col == WIDTH - 1)):
        ptr = Flags + page * (STRIDE // 4)
        flags = tl.load(ptr)
        if flags >= 2:
            # The uniform verify reservation is contiguous within each page.
            # A ragged verify wrote only its leading rows, using the same IDs.
            first_col = tl.maximum(0, col - last_row)
            first_row = last_row - (col - first_col)
            accepted = tl.load(Commit + i // WIDTH)
            kept = first_row + tl.minimum(tl.maximum(accepted - first_col, 0), col - first_col + 1)
            tl.device_assert(kept < (flags & COUNT_MASK), "kv12s: unwritten row in accepted prefix")
            slot = flags >> SHIFT
            tl.store(ptr, tl.where(kept > 0, (slot << SHIFT) | (kept + 1), 0))
            if kept == 0:
                tl.atomic_xchg(Locks + slot, 0, sem="release")


@triton.jit
def _release(Ids, Flags, Refs, StageLocks, OverflowLocks, NPAGES: tl.constexpr, H: tl.constexpr,
             STRIDE: tl.constexpr, SHIFT: tl.constexpr, STAGES: tl.constexpr, OVERFLOW: tl.constexpr,
             RECYCLE: tl.constexpr, BLOCK: tl.constexpr):
    page = tl.load(Ids + tl.program_id(0)).to(tl.int64)
    if (page > 0) & (page < NPAGES):
        ptr = Flags + page * (STRIDE // 4)
        if RECYCLE:
            flags = tl.load(ptr)
            tl.store(ptr, 0)
            if flags >= 2:
                slot = flags >> SHIFT
                tl.atomic_xchg(StageLocks + slot, 0, sem="release", mask=slot < STAGES)
            elif flags == 1:
                lane = tl.arange(0, BLOCK)
                ref = tl.load(Refs + page * (STRIDE // 4) + lane, lane < H * 2, other=0)
                tl.atomic_xchg(OverflowLocks + ref - 1, 0, sem="release", mask=(ref > 0) & (ref <= OVERFLOW))
        else:
            tl.store(ptr, 0)


class DeviceLayer(Layer):
    """The same envelope as the reference writer, with device-owned side slots."""

    def __init__(self, raw, heads, page_stride, offset, staging_pages, overflow_heads):
        super().__init__(raw, heads, page_stride, offset, staging_pages, overflow_heads)
        self.stage_locks = torch.zeros(staging_pages, dtype=torch.int32, device=self.device)
        self.overflow_locks = torch.zeros(overflow_heads, dtype=torch.int32, device=self.device)
        self.gather_workspace = None

    def write(self, locations, k, v, *, tentative=False):
        loc = locations.reshape(-1).contiguous()
        if not loc.numel():
            return
        k = k.reshape(-1, self.heads, DIM).to(torch.bfloat16).view(torch.int16)
        v = v.reshape_as(k).to(torch.bfloat16).view(torch.int16)
        if (k.shape[0] != loc.numel() or k.stride(-1) != 1 or v.stride(-1) != 1
                or k.device != self.device or v.device != self.device or loc.device != self.device
                or loc.dtype not in (torch.int32, torch.int64)):
            raise ValueError("inkling_kv12s: invalid device scatter geometry")
        grid = (loc.numel(),)
        _reserve[grid](loc, self.planes[-1], self.stage_locks, loc.numel(), self.pages,
                       PAGE, self.page_stride, self.stage.shape[0], STAGE_SHIFT, STAGE_MASK, tentative,
                       num_warps=1)
        _scatter[(loc.numel(), 2)](
            loc, k, v, self.planes[-1], self.stage, loc.numel(), self.pages,
            PAGE, self.heads, DIM, self.page_stride, *k.stride()[:2], *v.stride()[:2],
            self.stage.shape[0], STAGE_SHIFT, tentative, triton.next_power_of_2(self.heads * DIM),
        )
        if not tentative:
            self._launch_pack(loc, k, v, promote=False)
        self._launch_publish(loc, tentative=tentative, promote=False)

    def _launch_pack(self, loc, k, v, *, promote):
        _pack[(loc.numel(), self.heads, 2)](
            loc, k, v, *self.planes, self.stage, self.overflow, self.overflow_locks,
            loc.numel(), self.pages, PAGE, self.heads, DIM, self.page_stride,
            *k.stride()[:2], *v.stride()[:2], self.overflow.shape[0],
            STAGE_SHIFT, STAGE_MASK, CAP, CHUNK, EXP_LO, promote, num_warps=8,
        )

    def _launch_publish(self, loc, *, tentative, promote):
        _publish[(loc.numel(),)](
            loc, self.planes[-1], self.stage_locks, loc.numel(), self.pages,
            PAGE, self.page_stride, STAGE_SHIFT, STAGE_MASK, tentative, promote, num_warps=1,
        )

    def commit(self, pages):
        pages = pages.reshape(-1).contiguous()
        if pages.numel():
            self._launch_pack(pages, self.stage, self.stage, promote=True)
            self._launch_publish(pages, tentative=False, promote=True)

    def settle(self, locations, commit_lens):
        if (locations.ndim != 2 or commit_lens.shape != (locations.shape[0],)
                or locations.device != self.device or commit_lens.device != self.device
                or locations.dtype not in (torch.int32, torch.int64)
                or commit_lens.dtype not in (torch.int32, torch.int64)):
            raise ValueError("inkling_kv12s: invalid verify settlement geometry")
        if not locations.numel():
            return
        loc = locations.contiguous()
        _trim[(loc.numel(),)](
            loc, commit_lens.contiguous(), self.planes[-1], self.stage_locks,
            loc.shape[1], self.pages, PAGE, self.page_stride, STAGE_SHIFT, STAGE_MASK, num_warps=1,
        )
        # One candidate per full page. A rejected candidate is either empty or
        # still staged with a shorter count. Already packed draft pages are skipped.
        pages = torch.where(loc.remainder(PAGE) == PAGE - 1, loc // PAGE, 0)
        self.commit(pages)

    def release(self, pages, *, recycle=True):
        pages = pages.reshape(-1).contiguous()
        if pages.numel():
            _release[(pages.numel(),)](
                pages, self.planes[-1], self.planes[4], self.stage_locks, self.overflow_locks,
                self.pages, self.heads, self.page_stride, STAGE_SHIFT,
                self.stage.shape[0], self.overflow.shape[0], recycle,
                triton.next_power_of_2(self.heads * 2), num_warps=1,
            )

    def clear(self):
        # Free space may currently be occupied by SWA/state. Stock zeros the
        # actual envelope when a FULL page is handed out.
        self.stage_locks.zero_()
        self.overflow_locks.zero_()

    def _snapshot_reference(self, *, words):
        """Build the host ownership mirror only at the synchronous snapshot API."""
        import heapq
        from .storage import EMPTY, PACKED, OpenPage

        ref = Layer.__new__(Layer)
        ref.__dict__.update(self.__dict__)
        flags = self.flags[words].tolist()
        ref.open = {}
        for page, flag in zip(words.tolist(), flags):
            if flag not in (EMPTY, PACKED):
                ref.open[page] = OpenPage(flag >> STAGE_SHIFT, (flag & STAGE_MASK) - 1)
        ref.spilled = {}
        ref._free_stage = (self.stage_locks == 0).nonzero().flatten().tolist()
        ref._free_overflow = (self.overflow_locks == 0).nonzero().flatten().tolist()
        heapq.heapify(ref._free_stage)
        heapq.heapify(ref._free_overflow)
        return ref

    def read_rows(self, locations):
        pages = self._ids(list(dict.fromkeys(loc // PAGE for loc in locations)))
        return self._snapshot_reference(words=pages).read_rows(locations)

    def prepare_write(self, locations, k, v, *, tentative=False):
        pages = self._ids(list(dict.fromkeys(loc // PAGE for loc in locations if loc > 0)))
        reference = self._snapshot_reference(words=pages)
        reference.prepare_write(locations, k, v, tentative=tentative)
        groups = {}
        for row, loc in enumerate(locations):
            if loc > 0:
                groups.setdefault(loc // PAGE, []).append(row)
        # Close whole pages in the same batches checked by the reference.
        # Put the final tails last so the direct full pages don't need slots.
        batches = ([], [])
        for rows in groups.values():
            rows.sort(key=locations.__getitem__)
            full = not tentative and locations[rows[-1]] % PAGE == PAGE - 1
            batches[0 if full else 1].append(rows)

        def publish():
            for groups in batches:
                for start in range(0, len(groups), PACK_BATCH):
                    rows = [row for group in groups[start:start + PACK_BATCH] for row in group]
                    ids = torch.tensor(rows, device=k.device, dtype=torch.long)
                    loc = self._ids([locations[row] for row in rows])
                    self.write(loc, k.index_select(0, ids).to(self.device),
                               v.index_select(0, ids).to(self.device), tentative=tentative)
        return publish


class RawLane:
    """The MTP draft's global lane, kept as raw BF16 inside the packed page envelope.

    Draft K/V escapes the packed exponent window far more often than the target's
    (live serving filled thousands of page-head overflow slots in minutes), and the
    draft proposal reads the whole prefix with noncausal FA4. A raw page-major lane
    costs 4096 B/token like stock and gives FA4 a strided (pages, 128, heads, 128)
    cache at the original physical page IDs: no gather and the stock arithmetic.
    Moves copy whole envelopes, so the lane needs no side-slot lifetime."""

    is_raw = True

    def __init__(self, raw, heads, page_stride, offset, pages):
        self.heads = heads
        self.device = raw.device
        self.pages = pages
        self.page_stride = page_stride
        size = raw_lane_bytes(heads)
        block = raw[:pages * page_stride].view(pages, page_stride)[:, offset:offset + size]
        kv = block.view(torch.bfloat16).unflatten(-1, (2, PAGE, heads, DIM))
        self.k, self.v = kv[:, 0], kv[:, 1]

    def write(self, locations, k, v, *, tentative=False):
        loc = locations.reshape(-1).to(torch.int64)
        if not loc.numel():
            return
        k = k.reshape(-1, self.heads, DIM).to(torch.bfloat16)
        v = v.reshape(-1, self.heads, DIM).to(torch.bfloat16)
        if k.shape[0] != loc.numel() or v.shape[0] != loc.numel() or loc.device != self.device:
            raise ValueError("inkling_kv12s: invalid raw lane scatter geometry")
        # Masked rows carry location 0, the sink page, as in the stock view.
        page, row = loc // PAGE, loc % PAGE
        self.k[page, row] = k
        self.v[page, row] = v

    def read_rows(self, locations):
        loc = torch.as_tensor(locations, dtype=torch.int64, device=self.device)
        page, row = loc // PAGE, loc % PAGE
        return torch.stack((self.k[page, row], self.v[page, row]), dim=1)

    def prepare_write(self, locations, k, v, *, tentative=False):
        loc = torch.as_tensor(locations, dtype=torch.int64, device=self.device)

        def publish():
            self.write(loc, k.to(self.device), v.to(self.device))
        return publish

    # The packed lanes' lifetime hooks: nothing to settle, release or recycle.
    def settle(self, locations, commit_lens):
        pass

    def commit(self, pages):
        pass

    def release(self, pages, *, recycle=True):
        pass

    def clear(self):
        pass
