"""CPU tests of inkling_moe2 (no torch, no sglang): parsing, table lookup, wrapping over stock / moe_tiles,
source checks and install(). Run: python3 -m unittest discover -s tests  (from the track dir)."""

import os
import sys
import tempfile
import types
import unittest

OVERLAY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "overlay")
sys.path.insert(0, OVERLAY)

import inkling_moe2  # noqa: E402
from inkling_moe2 import tiles  # noqa: E402
from inkling_kernels import moe_tiles  # noqa: E402

SHIPPED = ("1@2304:swap_ab=1,tile_n=32,tile_k=256;1@1152:swap_ab=1,tile_n=32,tile_k=256;"
           "8@2304:swap_ab=1,tile_n=32,tile_k=256;8@1152:swap_ab=1,tile_n=32,tile_k=256;"
           "64@2304:swap_ab=0,tile_n=64,tile_k=256;64@1152:swap_ab=1,tile_n=32,tile_k=256")


def stock_tiles(tokens, width):
    if tokens <= 128:
        return True, 32, 128
    if tokens <= 1536:
        return True, 32, 256
    if tokens <= 3072:
        return True, 64, 256
    return (False, 128, 256) if width >= 2048 else (True, 128, 128)


def fake_module():
    m = types.ModuleType("fake_grouped")
    m._tiles = stock_tiles
    m.mxfp8_expert_matmul = lambda *a: ("stock", a)
    return m


class Parse(unittest.TestCase):
    def test_nvfp4(self):
        r = tiles.parse_nvfp4("65-128:swap_ab=1,tile_n=32,tile_k=256; 129-256@2304:swap_ab=0,tile_n=64,tile_k=256")
        self.assertEqual(r[0], (129, 256, 2304, (False, 64, 256)))  # width-specific first
        self.assertEqual(r[1], (65, 128, None, (True, 32, 256)))
        self.assertEqual(tiles.parse_nvfp4(""), [])

    def test_nvfp4_rejects(self):
        for bad in ("65-128:tile_n=32", "65-128:swap_ab=2,tile_n=32,tile_k=256", "65-128:swap_ab=1,tile_n=256,tile_k=256",
                    "128-65:swap_ab=1,tile_n=32,tile_k=256", "65:swap_ab=1,tile_n=32,tile_k=256", "65-128",
                    "65-128:swap_ab=1,tile_n=32,tile_k=64"):
            with self.assertRaises(ValueError, msg=bad):
                tiles.parse_nvfp4(bad)

    def test_mxfp8(self):
        r = tiles.parse_mxfp8("513-1536@2304x1152:tile_m=64,tile_n=128;513-1536:tile_m=32,tile_n=128")
        self.assertEqual(r[0], (513, 1536, (2304, 1152), (64, 128)))
        self.assertEqual(r[1], (513, 1536, None, (32, 128)))
        for bad in ("1-8:tile_m=16,tile_n=128", "1-8:tile_m=64", "1-8:tile_m=64,tile_n=128,x=1"):
            with self.assertRaises(ValueError, msg=bad):
                tiles.parse_mxfp8(bad)


class Defaults(unittest.TestCase):
    def test_defaults_parse_and_extend_shipped(self):
        nv = tiles.parse_nvfp4(inkling_moe2.DEFAULT_NVFP4)
        mx = tiles.parse_mxfp8(inkling_moe2.DEFAULT_MXFP8)
        m = fake_module()
        moe_tiles.patch(m, moe_tiles.parse(SHIPPED))
        shipped = m._tiles
        tiles.patch(m, nv, mx)
        for t in (1, 8):  # c1 untouched
            for w in (1152, 2304):
                self.assertEqual(m._tiles(t, w), shipped(t, w))
        for t in (16, 32, 64, 96, 128):
            self.assertEqual(m._tiles(t, 2304), (True, 32, 256))
            self.assertEqual(m._tiles(t, 1152), (True, 32, 256))
        for t in (192, 512, 4096):  # stock above 128 tokens
            for w in (1152, 2304):
                self.assertEqual(m._tiles(t, w), stock_tiles(t, w))
        self.assertEqual(tiles.lookup(mx, 256, (2304, 1152)), None)
        self.assertEqual(tiles.lookup(mx, 512, (2304, 1152)), (32, 128))
        self.assertEqual(tiles.lookup(mx, 512, (2304, 2304)), None)
        self.assertEqual(tiles.lookup(mx, 1536, (2304, 2304)), (32, 128))
        self.assertEqual(tiles.lookup(mx, 2048, (2304, 1152)), None)


class Wrap(unittest.TestCase):
    def test_over_stock(self):
        m = fake_module()
        nv = tiles.parse_nvfp4("65-128:swap_ab=1,tile_n=32,tile_k=256;129-192@2304:swap_ab=0,tile_n=64,tile_k=256")
        self.assertEqual(tiles.patch(m, nv, []), ["nvfp4 over stock"])
        self.assertEqual(m._tiles(128, 2304), (True, 32, 256))
        self.assertEqual(m._tiles(65, 1152), (True, 32, 256))
        self.assertEqual(m._tiles(192, 2304), (False, 64, 256))
        self.assertEqual(m._tiles(192, 1152), stock_tiles(192, 1152))  # width rule does not match w2
        self.assertEqual(m._tiles(64, 2304), stock_tiles(64, 2304))    # outside every range -> delegate
        self.assertEqual(m._tiles(4096, 2304), stock_tiles(4096, 2304))
        self.assertEqual(tiles.patch(m, nv, []), [])                    # idempotent
        self.assertIs(m._tiles.__wrapped__, stock_tiles)

    def test_over_moe_tiles(self):
        m = fake_module()
        moe_tiles.patch(m, moe_tiles.parse(SHIPPED))
        shipped = m._tiles
        nv = tiles.parse_nvfp4("65-128:swap_ab=1,tile_n=32,tile_k=256")
        self.assertEqual(tiles.patch(m, nv, []), ["nvfp4 over moe_tiles"])
        for t in (1, 8, 32, 64):
            for w in (1152, 2304):
                self.assertEqual(m._tiles(t, w), shipped(t, w))
        self.assertEqual(m._tiles(64, 2304), (False, 64, 256))
        self.assertEqual(m._tiles(128, 2304), (True, 32, 256))
        self.assertEqual(m._tiles(192, 2304), stock_tiles(192, 2304))

    def test_mxfp8_wrapped_only(self):
        m = fake_module()
        stock = m.mxfp8_expert_matmul
        self.assertEqual(tiles.patch(m, [], tiles.parse_mxfp8("513-1536:tile_m=64,tile_n=128")), ["mxfp8"])
        self.assertIs(m.mxfp8_expert_matmul.__wrapped__, stock)
        self.assertIs(m._tiles, stock_tiles)


class Install(unittest.TestCase):
    def _tree(self, drop=None):
        d = tempfile.mkdtemp()
        files = {
            "kernels/ops/gemm/inkling2_grouped_sm120.py": "\n".join(tiles.SOURCE_NEEDLES),
            "srt/models/inkling_common/inkling2_moe.py":
                "grouped_sm120.nvfp4_expert_matmul(\ngrouped_sm120.mxfp8_expert_matmul(\ntokens=tokens,\n",
        }
        for rel, text in files.items():
            if drop and drop in text:
                text = text.replace(drop, "")
            os.makedirs(os.path.dirname(os.path.join(d, rel)), exist_ok=True)
            with open(os.path.join(d, rel), "w") as f:
                f.write(text)
        return d

    def setUp(self):
        inkling_moe2.STATE.update(installed=False, levers=(), applied=[])
        self.env = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("INKLING_MOE2")}

    def tearDown(self):
        for k in [k for k in os.environ if k.startswith("INKLING_MOE2")]:
            del os.environ[k]
        os.environ.update(self.env)
        inkling_moe2.STATE.update(installed=False, levers=(), applied=[])

    def test_off(self):
        self.assertEqual(inkling_moe2.install(root=self._tree()), ())

    def test_unknown(self):
        os.environ["INKLING_MOE2"] = "tiles,foo"
        with self.assertRaises(RuntimeError):
            inkling_moe2.install(root=self._tree())

    def test_changed_target(self):
        os.environ["INKLING_MOE2"] = "tiles"
        os.environ["INKLING_MOE2_NVFP4"] = "65-128:swap_ab=1,tile_n=32,tile_k=256"
        with self.assertRaises(RuntimeError):
            inkling_moe2.install(root=self._tree(drop="scale_granularity_mnk=(1, 1, 32),"))

    def test_empty_tables(self):
        os.environ["INKLING_MOE2"] = "tiles"
        os.environ["INKLING_MOE2_NVFP4"] = "none"
        os.environ["INKLING_MOE2_MXFP8"] = "none"
        with self.assertRaises(RuntimeError):
            inkling_moe2.install(root=self._tree())

    def test_hook(self):
        os.environ["INKLING_MOE2"] = "tiles"
        os.environ["INKLING_MOE2_NVFP4"] = "65-128:swap_ab=1,tile_n=32,tile_k=256"
        os.environ["INKLING_MOE2_MXFP8"] = "none"
        m = fake_module()
        sys.modules[tiles.TARGET_MODULE] = m  # already imported -> the callback runs at install
        try:
            self.assertEqual(inkling_moe2.install(root=self._tree()), ("tiles",))
        finally:
            del sys.modules[tiles.TARGET_MODULE]
        self.assertEqual(inkling_moe2.STATE["applied"], ["nvfp4 over stock"])
        self.assertEqual(m._tiles(100, 2304), (True, 32, 256))


if __name__ == "__main__":
    unittest.main()
