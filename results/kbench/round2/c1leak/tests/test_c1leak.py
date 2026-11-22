"""CPU tests of the c1leak track's Python side (torch CPU, no GPU): inkling_pico12 `stack1` and inkling_draftbytes
`w8gemv` option plumbing. Run from overlay/ with this file's dir and tests/pico12 on sys.path:

    PYTHONPATH=<overlay>:<overlay>/tests/pico12 python3 -m unittest test_c1leak -v
"""

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.environ.get("OVERLAY", "/ov")
for p in (OVERLAY, os.path.join(OVERLAY, "tests", "pico12")):
    if p not in sys.path:
        sys.path.insert(0, p)

import inkling_pico12  # noqa: E402
from fake_ext import FakeExt  # noqa: E402
from inkling_pico12 import ops  # noqa: E402


class FakeRange:
    """Stand-in of the stack1 range extension: the call contract of csrc/pico12_range.cu, reference math."""

    def __init__(self, ext):
        self.ext, self.calls = ext, []

    def _check(self, x, buf, n, k, out, raw_tiles):
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k and x.stride(1) == 1
        assert x.shape[0] == 1 or x.stride(0) == k
        assert out.dtype == torch.bfloat16 and out.dim() == 2 and out.shape == (x.shape[0], n) and out.stride(1) == 1
        assert out.shape[0] == 1 or out.stride(0) == n
        assert raw_tiles >= 0

    def gemv_range(self, x, buf, n, k, out, raw_tiles, mt, wk, pdl, lo, hi, inside):
        self._check(x, buf, n, k, out, raw_tiles)
        assert 1 <= x.shape[0] <= 16 and mt in (1, 2)
        r = 8 * mt
        assert 0 <= lo < hi <= n and lo % r == 0 and hi % r == 0 and (inside or n % r == 0)
        w = FakeExt.weight_of(buf, n, k)
        y = (x.float() @ w.float().T).to(torch.bfloat16)
        cols = torch.zeros(n, dtype=torch.bool)
        cols[lo:hi] = True
        if not inside:
            cols = ~cols
        out[:, cols] = y[:, cols]
        self.calls.append(("range", x.shape[0], inside, mt))

    def gemv_stacked(self, x, buf, n, k, out, raw_tiles, m, wk, pdl, lo, hi):
        self._check(x, buf, n, k, out, raw_tiles)
        assert 1 <= m <= 8 and x.shape[0] == 3 * m
        w = FakeExt.weight_of(buf, n, k)
        y = (x.float() @ w.float().T).to(torch.bfloat16)
        out[:m, :lo] = y[:m, :lo]
        out[:m, hi:] = y[:m, hi:]
        out[m:, lo:hi] = y[m:, lo:hi]
        self.calls.append(("stacked", m))


SPLITS = [24, 8, 8, 8, 16, 8]  # q, k, v, r, gate, sinks (multiples of 8)
N, K = sum(SPLITS), 64


class Stack1(unittest.TestCase):
    def setUp(self):
        self.ext = FakeExt()
        ops.EXT, ops.RANGE_EXT = self.ext, FakeRange(self.ext)
        ops.SET.require_cuda = False
        ops.SET.max_m = ops.SET.stacked_max_m = 16
        ops.SET.wk, ops.SET.pdl = 1, True
        self.w = (torch.randn(N, K) * 0.1).to(torch.bfloat16)
        self.full = ops.encode(self.w, "qkvr")
        self.lo, self.hi = ops.kv_bounds(SPLITS)

    def tearDown(self):
        ops.SET.stack1, ops.SET.stack1_fuse, ops.SET.stack1_kv_mt = False, False, 1
        ops.RANGE_EXT = None

    def kept(self, o3, m):
        lo, hi, klo = self.lo, self.hi, self.lo + SPLITS[1]
        q = torch.cat([o3[:m, :lo], o3[:m, hi:]], 1)
        return torch.cat([q.flatten(), o3[m : 2 * m, lo:klo].flatten(), o3[2 * m :, klo:hi].flatten()])

    def test_no_kv_pack_needed(self):
        ops.SET.stack1 = True
        self.assertFalse(ops.stacked_needs_kv(16))
        ops.SET.stack1 = False
        self.assertTrue(ops.stacked_needs_kv(16))

    def test_values_and_calls(self):
        plan = ops.QkvrPlan(self.full, None, self.lo, self.hi)
        for fuse in (False, True):
            for kv_mt in (1, 2):
                for m in (1, 3, 5, 6, 8, 9, 12, 16):
                    ops.SET.stack1, ops.SET.stack1_fuse, ops.SET.stack1_kv_mt = True, fuse, kv_mt
                    x = torch.randn(3 * m, K).to(torch.bfloat16)
                    src = (x[:m], x[m : 2 * m], x[2 * m :])
                    self.assertTrue(ops.stacked_eligible(src, plan))
                    ops.RANGE_EXT.calls.clear()
                    self.ext.reset()
                    out = ops.stacked_project(x, m, plan)
                    ref = (x.float() @ self.w.float().T).to(torch.bfloat16)
                    self.assertTrue(torch.equal(self.kept(out, m), self.kept(ref, m)), (fuse, kv_mt, m))
                    calls = ops.RANGE_EXT.calls
                    if 3 * m <= 16:
                        self.assertEqual(calls, [])
                        self.assertEqual(len(self.ext.calls), 1)
                    elif fuse and m <= 8:
                        self.assertEqual(calls, [("stacked", m)])
                    else:
                        want_kv = [r1 - r0 for r0, r1 in ((r, min(3 * m, r + 16)) for r in range(m, 3 * m, 16))]
                        self.assertEqual(calls[0][:3], ("range", m, False))
                        self.assertEqual([c[1] for c in calls[1:]], want_kv)
                        self.assertTrue(all(c[2] for c in calls[1:]))
                        self.assertEqual(len(self.ext.calls), 0)  # no plain gemv, no kv pack

    def test_ineligible_without_stack1(self):
        plan = ops.QkvrPlan(self.full, None, self.lo, self.hi)
        x = torch.randn(24, K).to(torch.bfloat16)
        ops.SET.stack1 = False
        self.assertFalse(ops.stacked_eligible((x[:8], x[8:16], x[16:]), plan))
        ops.SET.stack1 = True
        self.assertTrue(ops.stacked_eligible((x[:8], x[8:16], x[16:]), plan))
        self.assertFalse(ops.stacked_eligible((x[:8], x[8:16], x[16:]), ops.QkvrPlan(self.full, None, 0, 0)))

    def test_range_mt(self):
        self.assertEqual(ops._range_mt(2, 16, 48), 2)
        self.assertEqual(ops._range_mt(2, 8, 48), 1)
        self.assertEqual(ops._range_mt(1, 8, 48), 1)


class Levers(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(inkling_pico12.parse_levers("dense,stack1"), ("dense", "stack1"))
        self.assertNotIn("stack1", inkling_pico12.parse_levers("all"))
        with self.assertRaises(RuntimeError):
            inkling_pico12.parse_levers("lmhead,stack1")
        cfg = inkling_pico12.config_from_env({})
        self.assertEqual((cfg["stack1_kv_mt"], cfg["stack1_fuse"]), (1, False))
        cfg = inkling_pico12.config_from_env({"INKLING_PICO12_STACK1_KV_MT": "2", "INKLING_PICO12_STACK1_FUSE": "1"})
        self.assertEqual((cfg["stack1_kv_mt"], cfg["stack1_fuse"]), (2, True))
        with self.assertRaises(ValueError):
            inkling_pico12.config_from_env({"INKLING_PICO12_STACK1_KV_MT": "3"})


class DraftOpts(unittest.TestCase):
    def test_w8gemv(self):
        from inkling_draftbytes import _opts

        o = _opts.parse("draftmx8,w8gemv", environ={})
        self.assertTrue(o.w8gemv)
        self.assertIn("w8gemv", o.describe())
        self.assertFalse(_opts.parse("draftmx8", environ={}).w8gemv)
        for bad in ("w8gemv", "draftmx8:row,w8gemv"):
            with self.assertRaises(_opts.OptionError):
                _opts.parse(bad, environ={})
        from unittest import mock

        with mock.patch.dict(os.environ, {"INKLING_DRAFT_W8GEMV_CFG": "1242"}):  # tunables read os.environ
            self.assertEqual(_opts.parse("draftmx8,w8gemv", environ={}).w8gemv_cfg, 1242)
        with mock.patch.dict(os.environ, {"INKLING_DRAFT_W8GEMV_CFG": "128"}):
            with self.assertRaises(_opts.OptionError):
                _opts.parse("draftmx8,w8gemv", environ={})

    def test_w8tile(self):
        from inkling_draftbytes import _opts

        self.assertTrue(_opts.parse("draftmx8:row,w8tile", environ={}).w8tile)
        self.assertIn("w8tile", _opts.parse("draftmx8,w8tile", environ={}).describe())
        with self.assertRaises(_opts.OptionError):
            _opts.parse("w8tile", environ={})

    def test_eligible_pack(self):
        from inkling_draftbytes import quant, w8gemv

        w = (torch.randn(64, 128) * 0.02).to(torch.bfloat16)
        self.assertTrue(w8gemv.eligible_pack(quant.quantize(w, "mx")))
        self.assertFalse(w8gemv.eligible_pack(quant.quantize(w, "row")))
        self.assertFalse(w8gemv.eligible_pack(quant.bf16_pack(w)))
        self.assertTrue(w8gemv.bias_ok(None, 64))
        self.assertTrue(w8gemv.bias_ok(torch.zeros(64, dtype=torch.bfloat16), 64))
        self.assertFalse(w8gemv.bias_ok(torch.zeros(64), 64))


if __name__ == "__main__":
    unittest.main()
