#!/usr/bin/env python3
"""GB10 check for packed-only KV. Run in the SGLang image with no server."""

import argparse
from contextlib import contextmanager
from itertools import accumulate
import math
import os
from pathlib import Path
import signal
from statistics import median
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "overlay"))

import torch

from inkling_kv12s.storage import DIM, EMPTY, PACKED, PAGE, STAGE_MASK, Layer, Layout

SCALE = DIM ** -0.5
HEADS_K = 8
HEADS_Q = 24
EXTENT = 1024
GIB = 2**30
BF16_KV_BYTES_PER_ROW = 2 * HEADS_K * DIM * 2


def equal_bits(label, left, right):
    storage_dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}
    left = left.contiguous().view(storage_dtype.get(left.dtype, left.dtype))
    right = right.contiguous().view(storage_dtype.get(right.dtype, right.dtype))
    if not torch.equal(left, right):
        delta = left != right
        raise AssertionError(f"{label}: {delta.count_nonzero().item():,} words differ out of {left.numel():,}")


def cold_us(fn, flush, reps):
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(reps):
        flush.add_(1)  # Read and write well beyond the GB10 L2 capacity.
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(1000 * start.elapsed_time(end))
    return median(samples)


def normal(shape, device, generator):
    return torch.randn(shape, dtype=torch.bfloat16, device=device, generator=generator)


def raw_cache(rows, device, generator):
    """Allocate only the oracle cache and fill it in bounded fixture chunks."""
    shape = (rows + PAGE, HEADS_K, DIM)
    k, v = (torch.empty(shape, dtype=torch.bfloat16, device=device) for _ in range(2))
    from inkling_kv12s.check_data import kv_values

    for cache in (k, v):
        cache[:PAGE].zero_()
        for begin in range(PAGE, rows + PAGE, 4096):
            end = min(rows + PAGE, begin + 4096)
            cache[begin:end] = kv_values(end - begin, HEADS_K, device, generator, real_like=True)
    # Force one nonzero-exponent overflow in K and one in V without Inf/NaN.
    k.view(torch.int16)[PAGE + 17, 0, :] = 114 << 7
    v.view(torch.int16)[2 * PAGE + 33, 2, :] = (131 << 7) | 53
    return k, v


def append(layer, k, v, begin, end, *, tentative=False):
    ids = list(range(PAGE + begin, PAGE + end))
    layer.write(ids, k[PAGE + begin:PAGE + end], v[PAGE + begin:PAGE + end], tentative=tentative)


def lengths(device, values):
    return torch.tensor(values, dtype=torch.int32, device=device)


def table_for(rows, device):
    return torch.arange(1, math.ceil(rows / PAGE) + 1, dtype=torch.int32, device=device)[None, :]


def print_timing(label, stock, packed, flush, reps, kv_rows):
    if reps > 0:
        stock_us = cold_us(stock, flush, reps)
        packed_us = cold_us(packed, flush, reps)
        kv_kbytes = kv_rows * BF16_KV_BYTES_PER_ROW / 1000
        print(f"{label:32} bitwise  cold L2 median ({reps} samples)  "
              f"stock {stock_us:10.1f} us ({kv_kbytes / stock_us:7.1f} GB/s)   "
              f"packed {packed_us:10.1f} us ({kv_kbytes / packed_us:7.1f} GB/s)   "
              f"{stock_us / packed_us:.3f}x",
              flush=True)
    else:
        print(f"{label:32} bitwise", flush=True)


def check_bits(device):
    """Exercise the device loader with every BF16 word, including NaN payloads."""
    from inkling_kv12s.attention import load_test_pages

    raw = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(2, 2, PAGE, 1, DIM)
    layer = Layer.standalone(3, 1, device=device)
    k, v = raw.transpose(0, 1).reshape(2, -1, 1, DIM).view(torch.bfloat16)
    append_locs = list(range(PAGE, 3 * PAGE))
    layer.write(append_locs[:PAGE], k[:PAGE], v[:PAGE])
    layer.write(append_locs[PAGE:], k[PAGE:], v[PAGE:], tentative=True)
    for block in (32, 64, 128):
        equal_bits(f"overflow/staging {block}", load_test_pages(layer, 2, block), raw)

    layer = Layer.standalone(3, 1, device=device)
    i = torch.arange(64, dtype=torch.int16, device=device)
    row_words = 0x3F80 + torch.arange(PAGE, dtype=torch.int16, device=device)
    words = row_words[None, None, :, None, None].expand_as(raw).clone()
    words[:, :, 63, 0, :64] = i | 0x7F80
    words[:, :, 64, 0, 0] = -32767
    k, v = words.transpose(0, 1).reshape(2, -1, 1, DIM).view(torch.bfloat16)
    layer.write(append_locs, k, v)
    if layer.spilled:
        raise AssertionError("64 escapes must fit the fixed stream")
    for block in (32, 64, 128):
        equal_bits(f"inline escapes {block}", load_test_pages(layer, 2, block), words)
    print("Triton reader                    bitwise (all BF16 patterns, overflow/staging, fixed escapes)",
          flush=True)


def report_compact(label, resources, spilling):
    print(f"{label} Triton resources {resources}", flush=True)
    if resources["spills"]:
        spilling.append(f"{label} ({resources['spills']} spills)")


def check_decode(rows, device, generator, flush, reps, spilling):
    from inkling_kv12s.attention import partials, prepare_partials
    from sglang.kernels.ops.attention.inkling2_attn import combine_gate_sink
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    k, v = raw_cache(rows, device, generator)
    layer = Layer.standalone(rows // PAGE + 1, HEADS_K, device=device)
    append(layer, k, v, 0, rows - PAGE)
    append(layer, k, v, rows - PAGE, rows, tentative=True)
    if not layer.spilled or len(layer.open) != 1:
        raise AssertionError("decode fixture did not exercise overflow and staging")
    kc, vc = (t.view(-1, PAGE, HEADS_K, DIM) for t in (k, v))
    cases = [(rows, 1), (rows, 8), (rows, 64)]
    if rows == 2048:
        cases.insert(0, (1024, 1))  # Stock's short-context 32-key decode tile.
    for span, qlen in cases:
        table, seq = table_for(span, device), lengths(device, [span])
        q = normal((qlen, HEADS_Q, DIM), device, generator)
        bias = normal((qlen, HEADS_Q, EXTENT), device, generator)
        gate = normal((qlen, HEADS_Q * DIM), device, generator)
        sinks = normal((qlen, HEADS_Q), device, generator)
        resources = {}

        def stock():
            return decode_partials(q, kc, vc, table, seq, bias, SCALE)

        label = f"compact {span // 1024}K q={qlen}"
        actual = partials(layer, q, table, seq, bias, SCALE, resources=resources)
        report_compact(label, resources, spilling)
        expected = stock()
        for part, a, b in zip(("output partial", "max", "sum"), actual, expected):
            equal_bits(f"{label} {part}", a, b)
        equal_bits(f"{label} gate/sink", combine_gate_sink(*actual, gate=gate, sinks=sinks),
                   combine_gate_sink(*expected, gate=gate, sinks=sinks))
        del actual, expected
        packed = None
        if reps:
            # Eager packed validation reads metadata back to the CPU. Keep
            # validation and packed output allocation outside the event pair.
            packed, _ = prepare_partials(layer, q, table, seq, bias, SCALE)
            packed()
            stock()
        print_timing(label, stock, packed, flush, reps, span)


def check_ragged_compact(device, generator, spilling):
    from inkling_kv12s.attention import partials
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    k, v = raw_cache(2 * PAGE, device, generator)
    layer = Layer.standalone(3, HEADS_K, device=device)
    append(layer, k, v, 0, PAGE)
    append(layer, k, v, PAGE, PAGE + 103, tentative=True)
    kc, vc = (t.view(-1, PAGE, HEADS_K, DIM) for t in (k, v))
    table = torch.tensor([[1], [2]], dtype=torch.int32, device=device)
    seq = lengths(device, [PAGE, 103])
    cu_q = lengths(device, [0, 8, 15])
    q = normal((15, HEADS_Q, DIM), device, generator)
    bias = normal((15, HEADS_Q, EXTENT), device, generator)
    # Both per-request stock calls use the same four 32-key splits as the
    # two-request ragged launch.
    resources = {}
    actual = partials(layer, q, table, seq, bias, SCALE, cu_q=cu_q, qmax=8, resources=resources)
    report_compact("ragged compact q=8/7", resources, spilling)
    for b, (start, end) in enumerate(((0, 8), (8, 15))):
        expected = decode_partials(q[start:end], kc, vc, table[b:b + 1], seq[b:b + 1], bias[start:end], SCALE)
        for a, expected_part in zip(actual, expected):
            equal_bits("ragged compact partial", a[:, start:end], expected_part)
    print("ragged compact partials         bitwise", flush=True)


def fa4_kwargs(q, layer, table, cu_q, seq_k, qmax, bias, *, mode, padded=False):
    """Mirror Inkling2's eager global BF16 paged FA4 calls on GB10."""
    splits = 1  # Both RadixAttention instances override the backend split policy.
    if padded and mode == "verify":
        from sglang.srt.models.inkling_common.inkling2_attn import wave_fill_splits
        from sglang.srt.utils import get_device_core_count

        batch = seq_k.numel()
        group = q.shape[1] // layer.heads
        tiles = batch * layer.heads * -(-q.shape[0] * group // (128 * batch))
        splits = wave_fill_splits(tiles, get_device_core_count(q.device.index))

    kwargs = dict(
        q=q, k_cache=layer, v_cache=layer, page_table=table, cu_seqlens_q=cu_q,
        cache_seqlens=seq_k, max_seqlen_q=qmax, rel_bias=bias, softmax_scale=SCALE,
        causal=True, window_size=(-1, -1), softcap=0.0, num_splits=splits, pack_gqa=padded,
        rel_bias_prep_cache={},
        return_softmax_lse=True, out=torch.empty_like(q, dtype=torch.float32),
    )
    if mode == "decode":
        # SM12x uses static max-K metadata. Scheduler metadata is FA3 only.
        kwargs.update(max_seqlen_k=int(seq_k.max().item()), scheduler_metadata=None)
    else:
        kwargs["cu_seqlens_k_new"] = torch.nn.functional.pad(seq_k.cumsum(0, dtype=torch.int32), (1, 0))
    return kwargs


def check_fa4(device, generator):
    from inkling_kv12s.attention import FA4Workspace
    from inkling_kv12s.patch import dispatch_packed_fa4
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.inkling2_attn import pad_head_groups

    k, v = raw_cache(6 * PAGE, device, generator)
    layer = Layer.standalone(7, HEADS_K, device=device)
    workspace = FA4Workspace(layer.pages, layer.heads, device)
    append(layer, k, v, 0, 5 * PAGE)
    append(layer, k, v, 5 * PAGE, 5 * PAGE + 39, tentative=True)
    kc, vc = (t.view(-1, PAGE, HEADS_K, DIM) for t in (k, v))
    # Nonconsecutive shared prefix and unused trailing columns. Page 4 and 5
    # are live but must not appear in this call's temporary BF16 gather.
    base_table = torch.tensor([[2, 1, 6, 0], [2, 3, 0, 0]], dtype=torch.int32, device=device)
    for mode, padded, query_lengths in (
        ("decode", False, (1, 1)),
        ("decode", True, (1,) * 8),
        ("verify", False, (3, 3)),
        ("verify", True, (32, 32)),
        ("extend", False, (96, 53)),
    ):
        label = f"stock FA4 {'padded' if padded else 'generic'} {mode}"
        batch = len(query_lengths)
        table = base_table.repeat(batch // 2, 1)
        split_verify = padded and mode == "verify"
        gathered_pages = 5  # Four active pages and the sink.
        if split_verify:
            # FA4 can normalize a requested three splits to two on four
            # gathered pages, versus three on the seven-page raw cache.
            table[1, 1] = table[0, 1]
            gathered_pages = 4
        seq = lengths(device, [2 * PAGE + 39, 2 * PAGE] * (batch // 2))
        cu_q = lengths(device, list(accumulate(query_lengths, initial=0)))
        tokens = sum(query_lengths)
        q = normal((tokens, HEADS_Q, DIM), device, generator)
        bias = normal((tokens, HEADS_Q, EXTENT), device, generator)
        if padded:
            q, bias = (pad_head_groups(t, HEADS_Q // HEADS_K, 4) for t in (q, bias))
        kwargs = fa4_kwargs(q, layer, table, cu_q, seq, max(query_lengths), bias, mode=mode, padded=padded)
        expected = flash_attn_with_kvcache(**dict(
            kwargs, k_cache=kc, v_cache=vc, out=torch.empty_like(kwargs["out"]), rel_bias_prep_cache={},
        ))

        def stock_on_gather(**call):
            replaced = {"k_cache", "v_cache", "page_table"}
            expected_keys = kwargs.keys() | {"max_seqlen_k"} if split_verify else kwargs.keys()
            if (call.keys() != expected_keys
                    or any(call[name] is not kwargs[name] for name in kwargs.keys() - replaced)):
                raise AssertionError(f"{label}: packed gather changed FA4 arguments")
            if split_verify and call["max_seqlen_k"] != kc.shape[0] * PAGE:
                raise AssertionError(f"{label}: packed gather changed FA4's implicit split-planning bound")
            if call["k_cache"].shape[0] != gathered_pages or call["v_cache"].shape[0] != gathered_pages:
                raise AssertionError(f"{label}: expected {gathered_pages} gathered pages including the sink")
            return flash_attn_with_kvcache(**call)

        actual = dispatch_packed_fa4(stock_on_gather, layer, kwargs)
        for part, a, b in zip(("output", "LSE"), actual[:2], expected[:2]):
            equal_bits(f"{label} {part}", a, b)
        layer.gather_workspace = workspace
        static = dispatch_packed_fa4(
            flash_attn_with_kvcache, layer, dict(kwargs, out=torch.empty_like(kwargs["out"]), rel_bias_prep_cache={})
        )
        layer.gather_workspace = None
        for part, a, b in zip(("output", "LSE"), static[:2], expected[:2]):
            equal_bits(f"{label} static gather {part}", a, b)
        print(f"{label:32} bitwise (split={kwargs['num_splits']}, pack_gqa={kwargs['pack_gqa']})", flush=True)


def check_prefill(prefix, chunk, device, generator, flush, reps, warps):
    from inkling_attn.ops import prefill_attention
    from inkling_attn.patch import use_visit
    from inkling_kv12s.patch import dispatch_packed_fa4
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache

    k, v = raw_cache(prefix + 2 * PAGE, device, generator)
    kc, vc = (t.view(-1, PAGE, HEADS_K, DIM) for t in (k, v))
    layer = Layer.standalone(kc.shape[0], HEADS_K, device=device)
    append(layer, k, v, 0, prefix)
    proj = normal((16, EXTENT), device, generator) * 0.1
    # Complete a staged page across two successive chunks and leave a new tail.
    begin = prefix
    for qlen in (PAGE - chunk, chunk + 13):
        end = begin + qlen
        append(layer, k, v, begin, end)
        q = normal((qlen, HEADS_Q, DIM), device, generator)
        r = normal((qlen, HEADS_Q, 16), device, generator)
        table, seq, cu_q = table_for(end, device), lengths(device, [end]), lengths(device, [0, qlen])
        kwargs = fa4_kwargs(q, layer, table, cu_q, seq, qlen, r, mode="extend")

        def stock():
            return prefill_attention(q, kc, vc, table, cu_q, seq, r, proj, SCALE, None, warps=warps)

        def unexpected_fallback(**call):
            raise AssertionError("the projected prefill case did not dispatch to the fused reader")

        from inkling_kv12s import patch as kv12s_patch

        def packed(min_rows=0):
            # 0: BF16 prefix scratch + BF16 kernel; above the chunk: in-kernel packed loader.
            saved = kv12s_patch._SCRATCH_MIN_TOKENS
            kv12s_patch._SCRATCH_MIN_TOKENS = min_rows
            try:
                with use_visit(r, proj):
                    return dispatch_packed_fa4(unexpected_fallback, layer, kwargs, warps=warps)
            finally:
                kv12s_patch._SCRATCH_MIN_TOKENS = saved

        actual, expected = packed(), stock()
        label = f"fused prefill {begin}+{qlen}"
        for part, a, b in zip(("output", "LSE"), actual, expected):
            equal_bits(f"{label} {part}", a, b)
        for part, a, b in zip(("output", "LSE"), packed(1 << 30), expected):
            equal_bits(f"{label} in-kernel loader {part}", a, b)
        bias = torch.einsum("thd,de->the", r, proj)
        fa4_out, fa4_lse, *_ = flash_attn_with_kvcache(**dict(
            kwargs, k_cache=kc, v_cache=vc, rel_bias=bias, out=torch.empty_like(kwargs["out"]),
        ))
        out_error = (actual[0] - fa4_out).abs().max().item()
        lse_error = (actual[1] - fa4_lse).abs().max().item()
        if not (math.isfinite(out_error) and math.isfinite(lse_error)):
            raise AssertionError(f"{label}: nonfinite comparison with stock FA4")
        print(f"{label} vs stock FA4 max abs output={out_error:.6g}, LSE={lse_error:.6g}", flush=True)
        print_timing(label, stock, packed, flush, reps, end)
        begin = end


def check_reader_guards(device, generator, warps, spilling):
    from inkling_attn.ops import packed_prefill_attention, prefill_attention
    from inkling_kv12s.attention import load_test_pages, partials
    from inkling_kv12s.storage import CAP, STAGE_SHIFT
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    layer = Layer.standalone(6, 1, staging_pages=2, overflow_heads=2, device=device)
    expected = torch.zeros((6, 2, PAGE, 1, DIM), dtype=torch.int16, device=device)
    expected[1, :, :17].fill_(0x3F80)
    expected[3].fill_(0x3F80)
    expected[3, :, 0, 0, 0] = 0
    expected[3, :, 0, 0, 1] = 0
    layer.stage.fill_(0x4000)
    tail = expected[1, 0, :17].view(torch.bfloat16)
    layer.write(list(range(PAGE, PAGE + 17)), tail, tail, tentative=True)
    wide = torch.full_like(expected[2, 0], 114 << 7).view(torch.bfloat16)
    layer.write(list(range(2 * PAGE, 3 * PAGE)), wide, wide)
    layer.planes[4][2, 0, 0], layer.planes[4][2, 0, 1] = -1, 3
    inline = expected[3, 0].clone()
    inline[0, 0, :2] = 114 << 7
    inline = inline.view(torch.bfloat16)
    layer.write(list(range(3 * PAGE, 4 * PAGE)), inline, inline)
    layer.planes[3][3, 0, 0, 0] = -1
    layer.planes[3][3, 0, 1, 1:] = CAP + 1
    layer.flags[4] = (layer.stage.shape[0] << STAGE_SHIFT) | 2
    layer.flags[5] = 255
    equal_bits("bounded descriptor loads", load_test_pages(layer, 5), expected[1:])

    q = normal((8, 3, DIM), device, generator)
    r = normal((8, 3, 16), device, generator)
    bias = normal((8, 3, EXTENT), device, generator)
    proj = normal((16, EXTENT), device, generator)
    seq, cu_q = lengths(device, [7 * PAGE]), lengths(device, [0, 8])
    table = torch.tensor([[1, 2, 3, 4, 5, -1, layer.pages]], dtype=torch.int32, device=device)
    stock_table = torch.tensor([[1, 2, 3, 4, 5, 0, 0]], dtype=torch.int32, device=device)
    kc, vc = (expected[:, i].contiguous().view(torch.bfloat16) for i in range(2))
    resources = {}
    actual = partials(layer, q, table, seq, bias, SCALE, resources=resources)
    report_compact("compact guarded q=8", resources, spilling)
    for a, b in zip(actual, decode_partials(q, kc, vc, stock_table, seq, bias, SCALE)):
        equal_bits("compact guarded loads", a, b)
    for a, b in zip(packed_prefill_attention(q, layer, table, cu_q, seq, SCALE, r=r, proj=proj, warps=warps),
                    prefill_attention(q, kc, vc, stock_table, cu_q, seq, r, proj, SCALE, None, warps=warps)):
        equal_bits("fused guarded loads", a, b)

    for bad_table, bad_offsets in ((table[:, :-1], cu_q), (table, lengths(device, [0, 9]))):
        for run in (
            lambda: partials(layer, q, bad_table, seq, bias, SCALE, cu_q=bad_offsets),
            lambda: packed_prefill_attention(q, layer, bad_table, bad_offsets, seq, SCALE, r=r, proj=proj),
        ):
            try:
                run()
            except ValueError:
                continue
            raise AssertionError("reader accepted out-of-bounds sequence or query metadata")
    print("compact and fused metadata guards passed", flush=True)


@contextmanager
def phase(label, device, peaks, min_free=0):
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    print(f"[{label}] start, device free {free / GIB:.2f}/{total / GIB:.2f} GiB", flush=True)
    if free < min_free:
        raise RuntimeError(f"{label} needs at least {min_free / GIB:.1f} GiB of free device memory")
    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    yield
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    peaks[0] = max(peaks[0], allocated)
    peaks[1] = max(peaks[1], reserved)
    print(f"[{label}] done in {time.monotonic() - start:.1f}s, "
          f"peak allocated {allocated / GIB:.3f} GiB, reserved {reserved / GIB:.3f} GiB", flush=True)


def check_device_writer_bits(device):
    from inkling_kv12s.attention import load_test_pages
    from inkling_kv12s.device import DeviceLayer

    patterns = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(2, 2, PAGE, 1, DIM)
    finite = torch.full((1, 2, PAGE, 1, DIM), 0x3F85, dtype=torch.int16, device=device)
    finite[0, 0, 31:33, 0, :32] = 0x7F81
    finite[0, 1, ::4, 0, 1] = -32767
    words = torch.cat((patterns, finite))
    k, v = words.transpose(0, 1).reshape(2, -1, 1, DIM).view(torch.bfloat16)
    layer = DeviceLayer.standalone(5, 1, overflow_heads=8, device=device)
    loc = torch.arange(PAGE, 4 * PAGE, device=device)
    # Direct full-page encode, tentative staging and promotion, and a page
    # that closes by appending to an existing staged prefix.
    layer.write(loc[:PAGE], k[:PAGE], v[:PAGE])
    layer.write(loc[PAGE:2 * PAGE], k[PAGE:2 * PAGE], v[PAGE:2 * PAGE], tentative=True)
    layer.commit(torch.tensor([2], device=device))
    layer.write(loc[2 * PAGE:-7], k[2 * PAGE:-7], v[2 * PAGE:-7])
    layer.write(loc[-7:], k[-7:], v[-7:])
    equal_bits("device writer bit patterns", load_test_pages(layer, 3), words)
    if layer.flags.tolist() != [EMPTY, PACKED, PACKED, PACKED, EMPTY]:
        raise AssertionError("device writer did not publish sealed pages")
    # A stock whole-envelope move on a schedule stream transfers overflow
    # references. Reusing the vacated ID on the forward stream must preserve
    # both the transferred overflow and the new staged row.
    words_by_page = layer.raw.view(layer.pages, layer.page_stride)
    forward = torch.cuda.current_stream(device)
    schedule = torch.cuda.Stream(device=device)
    schedule.wait_stream(forward)
    with torch.cuda.stream(schedule):
        words_by_page[4].copy_(words_by_page[1])
        layer.release(loc[:1] // PAGE, recycle=False)
    forward.wait_stream(schedule)
    layer.write(loc[:1], k[-1:], v[-1:])
    gathered = load_test_pages(layer, 4)
    equal_bits("moved overflow destination", gathered[3], words[0])
    equal_bits("reused move origin", gathered[0, :, 0], words[2, :, -1])
    layer.release(torch.tensor([1, 2, 3, 4], device=device))
    if layer.stage_locks.any().item() or layer.overflow_locks.any().item() or layer.flags.any().item():
        raise AssertionError("device writer leaked a side slot after release")
    print("device writer                    bitwise (all BF16 words and inline escapes)", flush=True)


def capture_forward(run, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        run()  # Compile and load kernels before capture.
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = run()
    return graph, out


def check_graph_decode(device, generator):
    from inkling_kv12s.attention import partials
    from inkling_kv12s.device import DeviceLayer
    from sglang.kernels.ops.attention.inkling2_attn import combine_gate_sink
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    layer = DeviceLayer.standalone(5, HEADS_K, device=device)
    k, v = raw_cache(4 * PAGE, device, generator)
    kc, vc = (t.view(-1, PAGE, HEADS_K, DIM) for t in (k, v))
    loc = torch.zeros(1, dtype=torch.int64, device=device)
    table = torch.zeros((1, 2), dtype=torch.int32, device=device)
    seq = lengths(device, [1])
    q = normal((1, HEADS_Q, DIM), device, generator)
    bias = normal((1, HEADS_Q, EXTENT), device, generator)
    gate = normal((1, HEADS_Q * DIM), device, generator)
    sinks = normal((1, HEADS_Q), device, generator)
    row_k, row_v = (torch.zeros_like(t[:1]) for t in (k, v))

    def run():
        layer.write(loc, row_k, row_v)
        parts = partials(layer, q, table, seq, bias, SCALE, device_metadata=True)
        return (*parts, combine_gate_sink(*parts, gate=gate, sinks=sinks))

    graph, actual = capture_forward(run, device)
    for page, prefix, offset in ((1, 2, 0), (3, PAGE - 1, 1)):
        base = page * PAGE
        layer.write(torch.arange(base, base + prefix, device=device), k[base:base + prefix], v[base:base + prefix])
        loc.fill_(base + prefix)
        seq.fill_(prefix + 1)
        table[0, 0] = page
        row_k.copy_(k[base + prefix:base + prefix + 1])
        row_v.copy_(v[base + prefix:base + prefix + 1])
        if offset:
            q.copy_(normal(q.shape, device, generator))
        graph.replay()
        expected = decode_partials(q, kc, vc, table, seq, bias, SCALE)
        for got, want in zip(actual, (*expected, combine_gate_sink(*expected, gate=gate, sinks=sinks))):
            equal_bits("captured q=1 replay", got, want)
    print("CUDA graph decode replay          bitwise (changed page IDs, lengths and Q/K/V)", flush=True)


def check_ragged_graph(device, generator):
    from inkling_kv12s.attention import partials
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    k, v = raw_cache(2 * PAGE, device, generator)
    layer = Layer.standalone(3, HEADS_K, device=device)
    append(layer, k, v, 0, PAGE)
    append(layer, k, v, PAGE, PAGE + 103, tentative=True)
    kc, vc = (cache.view(-1, PAGE, HEADS_K, DIM) for cache in (k, v))
    table = torch.tensor([[1], [2]], dtype=torch.int32, device=device)
    seq = lengths(device, [PAGE, 103])
    cu_q = lengths(device, [0, 8, 15])
    q = normal((15, HEADS_Q, DIM), device, generator)
    bias = normal((15, HEADS_Q, EXTENT), device, generator)

    def run():
        return partials(layer, q, table, seq, bias, SCALE, cu_q=cu_q, qmax=8, device_metadata=True)

    graph, actual = capture_forward(run, device)
    for cut, second_length in ((8, 103), (7, 91)):
        cu_q[1] = cut
        seq[1] = second_length
        graph.replay()
        for b, (start, end) in enumerate(((0, cut), (cut, 15))):
            expected = decode_partials(q[start:end], kc, vc, table[b:b + 1], seq[b:b + 1], bias[start:end], SCALE)
            for got, want in zip(actual, expected):
                equal_bits("captured ragged verify replay", got[:, start:end], want)
    print("CUDA graph ragged verify replay   bitwise (changed offsets and K lengths)", flush=True)


def check_speculative_graph(device, generator):
    from inkling_kv12s.attention import FA4Workspace, partials
    from inkling_kv12s.device import DeviceLayer
    from inkling_kv12s.patch import dispatch_packed_fa4
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache
    from sglang.kernels.ops.attention.inkling2_decode_attention_sm90 import decode_partials

    pages, width = 5, 8
    heads = (HEADS_K, 2)
    layouts = tuple(Layout(h).size for h in heads)
    stride = sum(layouts)
    raw = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
    target, draft = (DeviceLayer(raw, h, stride, offset, 4, 8)
                     for h, offset in zip(heads, (0, layouts[0])))
    draft.gather_workspace = FA4Workspace(pages, HEADS_K, device)
    prefix = PAGE - 3
    page_ids = (3, 1)
    table = torch.zeros((1, 2), dtype=torch.int32, device=device)
    seq = lengths(device, [width])
    cu_q = lengths(device, [0, width])
    loc = torch.zeros((1, width), dtype=torch.int64, device=device)
    accept = lengths(device, [0])
    cols = torch.arange(width, device=device)
    target_q = normal((width, HEADS_Q, DIM), device, generator)
    bias = normal((width, HEADS_Q, EXTENT), device, generator)
    draft_q = normal((width, 2 * draft.heads, DIM), device, generator)
    ta = tuple(normal((width, target.heads, DIM), device, generator) for _ in range(2))
    proposal = tuple(normal((width, draft.heads, DIM), device, generator) for _ in range(2))
    injected = tuple(normal((width, draft.heads, DIM), device, generator) for _ in range(2))
    kwargs = dict(q=draft_q, k_cache=draft, v_cache=draft, cache_seqlens=seq, page_table=table,
                  cu_seqlens_q=cu_q, max_seqlen_q=width, softmax_scale=SCALE, causal=False,
                  window_size=(-1, -1), softcap=0.0, num_splits=0, pack_gqa=True)

    def run():
        draft.write(loc, *proposal, tentative=True)
        proposed = dispatch_packed_fa4(flash_attn_with_kvcache, draft, kwargs)
        target.write(loc, *ta, tentative=True)
        verified = partials(target, target_q, table, seq, bias, SCALE, device_metadata=True)
        mask = torch.where(cols[None, :] < accept[:, None], loc, 0)
        draft.write(mask, *injected)
        return proposed, verified

    graph, (proposed, verified) = capture_forward(run, device)
    table.copy_(torch.tensor([page_ids], dtype=torch.int32, device=device))
    target_raw, draft_raw = (
        tuple(torch.zeros((pages * PAGE, h, DIM), dtype=torch.bfloat16, device=device) for _ in range(2))
        for h in heads
    )
    start = page_ids[0] * PAGE
    initial = torch.arange(start, start + prefix, device=device)
    history = list(range(start, start + prefix))
    for lane, cache in ((target, target_raw), (draft, draft_raw)):
        for source in cache:
            source[start:start + prefix].copy_(normal((prefix, lane.heads, DIM), device, generator))
        lane.write(initial, *(source[start:start + prefix] for source in cache))

    for accepted, expect_first, expect_next in ((2, 127, 0), (1, PAGE, 0), (8, PAGE, 8)):
        seq.fill_(prefix + width)
        positions = torch.arange(prefix, prefix + width, device=device)
        physical = table[0, positions // PAGE].long() * PAGE + positions % PAGE
        loc.copy_(physical.view_as(loc))
        accept.fill_(accepted)
        for values in (*ta, *proposal, *injected):
            values.copy_(normal(values.shape, device, generator))
        for cache, inputs in ((target_raw, ta), (draft_raw, proposal)):
            for dest, values in zip(cache, inputs):
                dest.index_copy_(0, physical, values)

        graph.replay()
        expected = decode_partials(
            target_q, *(t.view(pages, PAGE, target.heads, DIM) for t in target_raw),
            table, seq, bias, SCALE,
        )
        for got, want in zip(verified, expected):
            equal_bits("captured DSpark target verify", got, want)
        kc, vc = (t.view(pages, PAGE, draft.heads, DIM) for t in draft_raw)
        equal_bits("captured draft global proposal", proposed,
                   flash_attn_with_kvcache(**dict(kwargs, k_cache=kc, v_cache=vc)))
        for dest, values in zip(draft_raw, injected):
            dest.index_copy_(0, physical[:accepted], values[:accepted])
        history.extend(physical[:accepted].tolist())
        prefix += accepted
        for lane, cache in ((target, target_raw), (draft, draft_raw)):
            lane.settle(loc, accept)
            flags = lane.flags[list(page_ids)].tolist()
            for flag, end in zip(flags, (expect_first, expect_next)):
                if end in (0, PAGE):
                    assert flag == (PACKED if end else EMPTY)
                else:
                    assert flag >= 2 and (flag & STAGE_MASK) - 1 == end
            exact = torch.stack([t[history] for t in cache], dim=1)
            equal_bits("accepted target/draft rows", lane.read_rows(history), exact)
            if lane.stage_locks.count_nonzero().item() != sum(0 < end < PAGE for end in (expect_first, expect_next)):
                raise AssertionError("speculative reject leaked a staging slot")
    print("DSpark verify and draft replay    bitwise (accept, cross-page reject, reuse)", flush=True)


def check_raw_draft_lane(device, generator):
    """The MTP draft lane is raw BF16 inside the packed envelope: writes land at
    (page, row), and stock FA4 on its strided view equals FA4 on a contiguous copy."""
    from inkling_kv12s.device import DeviceLayer, RawLane
    from inkling_kv12s.storage import raw_lane_bytes
    from sglang.kernels.ops.attention.flash_attention_v4_sm120 import flash_attn_with_kvcache

    pages, heads = 6, HEADS_K
    packed = Layout(heads).size
    stride = packed + raw_lane_bytes(heads)
    raw = torch.zeros(pages * stride, dtype=torch.uint8, device=device)
    target = DeviceLayer(raw, heads, stride, 0, 4, 8)
    lane = RawLane(raw, heads, stride, packed, pages)
    rows = torch.arange(PAGE, 5 * PAGE + 41, device=device)
    k = normal((rows.numel(), heads, DIM), device, generator)
    v = normal((rows.numel(), heads, DIM), device, generator)
    tk = normal((rows.numel(), heads, DIM), device, generator)
    target.write(rows, tk, tk)
    lane.write(rows, k, v)
    masked = torch.where(rows < 2 * PAGE + 5, rows, 0)
    lane.write(masked, k, v)  # masked commit rows go to the sink page only
    equal_bits("raw draft lane rows", lane.read_rows(rows.tolist()), torch.stack((k, v), dim=1))
    equal_bits("packed target beside raw lane", target.read_rows(rows.tolist()), torch.stack((tk, tk), dim=1))
    kc = torch.zeros((pages, PAGE, heads, DIM), dtype=torch.bfloat16, device=device)
    vc = torch.zeros_like(kc)
    kc.view(-1, heads, DIM)[rows] = k
    vc.view(-1, heads, DIM)[rows] = v
    kc[0], vc[0] = lane.k[0], lane.v[0]
    table = torch.tensor([[1, 2, 3, 4, 5], [4, 2, 0, 0, 0]], dtype=torch.int32, device=device)
    seq = lengths(device, [4 * PAGE + 41, PAGE + 9])
    cu_q = lengths(device, [0, 8, 16])
    q = normal((16, HEADS_Q, DIM), device, generator)
    call = dict(q=q, page_table=table, cache_seqlens=seq, cu_seqlens_q=cu_q, max_seqlen_q=8,
                softmax_scale=SCALE, causal=False, window_size=(-1, -1), softcap=0.0)
    got = flash_attn_with_kvcache(k_cache=lane.k, v_cache=lane.v, **call)
    want = flash_attn_with_kvcache(k_cache=kc, v_cache=vc, **call)
    got, want = (x[0] if isinstance(x, tuple) else x for x in (got, want))
    equal_bits("stock FA4 on the strided raw draft lane", got, want)
    print("raw MTP draft lane                bitwise (writes, sink mask, strided FA4)", flush=True)


def run_checks(args):
    import triton
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"Torch {torch.__version__}, Triton {triton.__version__}, {torch.cuda.get_device_name(device)}", flush=True)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1, 8 * GIB / total), device)
    generator = torch.Generator(device=device).manual_seed(1221)
    peaks = [0, 0]
    spilling = []
    large_reps = args.reps if args.reps is not None else 20
    with torch.inference_mode():
        with phase("bits", device, peaks):
            check_bits(device)
            check_device_writer_bits(device)
        if not args.bits_only:
            with phase("reader guards", device, peaks):
                check_reader_guards(device, generator, args.warps, spilling)
            flush = torch.zeros(128 * 1024 * 1024 // 4, dtype=torch.int32, device=device) if (
                args.reps or args.prefill_reps or (args.large and large_reps)
            ) else None
            if flush is not None:
                print("Timing GB/s uses one logical BF16 K/V pass for both paths.", flush=True)
            with phase("compact decode and verify", device, peaks):
                check_decode(2 * 1024, device, generator, flush, args.reps or 0, spilling)
                check_ragged_compact(device, generator, spilling)
            with phase("CUDA graph replay and speculative commit", device, peaks):
                check_graph_decode(device, generator)
                check_ragged_graph(device, generator)
                check_speculative_graph(device, generator)
                check_raw_draft_lane(device, generator)
            with phase("stock FA4 gather", device, peaks):
                check_fa4(device, generator)
            with phase("fused prefill", device, peaks):
                check_prefill(4 * PAGE, 64, device, generator, flush, args.prefill_reps, args.warps)
            if args.large:
                for rows in (32 * 1024, 128 * 1024):
                    with phase(f"compact {rows // 1024}K", device, peaks, min_free=3 * GIB):
                        check_decode(rows, device, generator, flush, large_reps, spilling)
                with phase("fused prefill 128K prefix", device, peaks, min_free=3 * GIB):
                    check_prefill(128 * 1024, 64, device, generator, flush, args.prefill_reps, args.warps)
    if spilling:
        message = "Packed compact kernels spilled registers: " + ", ".join(spilling)
        if args.strict_spills:
            raise AssertionError(message)
        print(f"WARNING: {message}", file=sys.stderr, flush=True)
    print(f"Packed KV GPU checks passed. Peak allocated {peaks[0] / GIB:.3f} GiB, "
          f"reserved {peaks[1] / GIB:.3f} GiB.", flush=True)


def run_process(command, env, timeout, label):
    if timeout <= 0:
        raise SystemExit(f"{label}: checker deadline expired")
    with subprocess.Popen(command, env=env, start_new_session=True) as process:
        try:
            status = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise SystemExit(f"{label} timed out after {timeout:.0f}s") from None
        if status:
            raise SystemExit(f"{label} exited with status {status}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="run small parity cases (default)")
    mode.add_argument("--bits-only", action="store_true", help="run only the Triton bit loader, without building CUDA")
    mode.add_argument("--large", action="store_true", help="also exercise 32K/128K contexts")
    parser.add_argument("--strict-spills", action="store_true", help="fail on any compact register spill")
    parser.add_argument("--reps", type=int, help="cold-L2 decode samples per implementation (default: 20 at 32K/128K)")
    parser.add_argument("--prefill-reps", type=int, default=0, help="cold-L2 prefill samples per implementation")
    parser.add_argument("--warps", type=int, choices=(4, 8), default=4, help="fused prefill warps per CTA")
    parser.add_argument("--timeout-seconds", type=int, default=570, help="whole checker deadline")
    parser.add_argument("--build-timeout-seconds", type=int, default=240, help="extension prebuild deadline")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.reps or 0, args.prefill_reps) < 0 or min(args.timeout_seconds, args.build_timeout_seconds) <= 0:
        parser.error("sample counts must be nonnegative and deadlines must be positive")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
        parser.exit(status=1, message="The packed KV GPU check requires GB10 (SM121).\n")
    if args.worker:
        run_checks(args)
        return

    deadline = time.monotonic() + args.timeout_seconds
    with tempfile.TemporaryDirectory(prefix="kv12s-attn-") as cache:
        env = dict(os.environ, INKLING_ATTN_CACHE=cache, PYTHONUNBUFFERED="1")
        # Test Triton's strict default, even when the parent shell exports this override.
        env.pop("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", None)
        env.pop("TRITON_INTERPRET", None)
        if not args.bits_only:
            print("[extension prebuild] start (fresh cache)", flush=True)
            run_process(
                [sys.executable, "-c", f"import sys; sys.path.insert(0, {sys.path[0]!r}); "
                 "from inkling_attn import build; sys.exit(build.main([]))"],
                env, min(args.build_timeout_seconds, deadline - time.monotonic()), "extension prebuild",
            )
            print("[extension prebuild] done", flush=True)
        run_process([sys.executable, __file__, *sys.argv[1:], "--worker"], env,
                    deadline - time.monotonic(), "GPU checker")


if __name__ == "__main__":
    main()
