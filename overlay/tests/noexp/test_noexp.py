"""CPU tests of inkling_noexp: switch / dependency checks, source check, activation condition, routing call shape."""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inkling_noexp as ne  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class Switch(unittest.TestCase):
    def test_off(self):
        self.assertFalse(ne.enabled({}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "0"}))

    def test_needs_vcap_remap(self):
        # without the verify cap or its dead-row remap NOEXP stays inactive (stock routing)
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1"}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x", "INKLING_VERIFY_CAP_REMAP": "0"}))
        self.assertTrue(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x"}))
        with self.assertRaises(RuntimeError):
            ne.enabled({"INKLING_NOEXP": "maybe", "INKLING_VERIFY_CAP": "policy:x"})
        for off in ("false", " OFF ", "none"):
            self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": off}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x", "INKLING_VERIFY_CAP_REMAP": " off "}))

    def test_sources(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "srt/models/inkling_common/inkling2_moe.py")
        os.makedirs(os.path.dirname(p))
        open(p, "w").write("\n".join(ne._NEEDLES))
        ne.check_sources(d)
        open(p, "w").write("\n".join(ne._NEEDLES[:-1]))
        with self.assertRaises(RuntimeError):
            ne.check_sources(d)


@unittest.skipIf(torch is None, "needs torch (run in the SGLang image)")
class Routing(unittest.TestCase):
    def _vcap(self, **kw):
        st = types.SimpleNamespace(enabled=True, in_verify_moe=True, capping=True,
                                   config=types.SimpleNamespace(remap=True), stride=8,
                                   src_index=torch.tensor([0, 1, 2, 0, 0, 0, 0, 0, 8, 9, 8, 8, 8, 8, 8, 8]))
        for k, v in kw.items():
            setattr(st, k, v)
        mod = types.ModuleType("inkling_vcap.state")
        mod.STATE = st
        sys.modules["inkling_vcap.state"] = mod
        return st

    def tearDown(self):
        sys.modules.pop("inkling_vcap.state", None)

    def test_live_mask(self):
        self._vcap()
        live = ne.live_rows(16)
        self.assertEqual(live.tolist(), [True] * 3 + [False] * 5 + [True] * 2 + [False] * 6)

    def test_inactive_cases(self):
        self.assertIsNone(ne.live_rows(16))                     # no vcap
        self._vcap(capping=False)
        self.assertIsNone(ne.live_rows(16))
        self._vcap(in_verify_moe=False)
        self.assertIsNone(ne.live_rows(16))
        self._vcap()
        self.assertIsNone(ne.live_rows(12))                     # not a multiple of the stride
        self.assertIsNone(ne.live_rows(24))                     # beyond src_index

    def test_route_call(self):
        self._vcap()
        seen = {}

        def route_metadata(flat, w, start, num_local, num_experts):
            seen.update(flat=flat.clone(), start=start, num_local=num_local, num_experts=num_experts)
            return "s2d", "d2s", "g", None, "off", None

        ids = torch.arange(16 * 8).reshape(16, 8) % 128
        out = ne.route_dead_free(route_metadata, ids, torch.ones(16, 8), ne.live_rows(16), 128)
        self.assertEqual(out, ("s2d", "d2s", "g", "off"))
        self.assertEqual((seen["start"], seen["num_local"], seen["num_experts"]), (0, 128, 129))
        flat = seen["flat"].reshape(16, 8)
        self.assertTrue(torch.equal(flat[0], ids[0].int()))
        self.assertTrue((flat[3] == 128).all() and (flat[15] == 128).all())
        self.assertEqual(flat.dtype, torch.int32)

    def test_patch_passthrough(self):
        calls = []
        mod = types.SimpleNamespace(route_metadata=None, ROUTE_MAX_IDS=4096,
                                    Inkling2Experts=type("E", (), {"_sm120_route": lambda self, i, w: calls.append(1)
                                                                   or "stock"}))
        self.assertTrue(ne.patch(mod))
        self.assertFalse(ne.patch(mod))
        e = mod.Inkling2Experts()
        e.num_global_experts = 128
        from unittest import mock

        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: False):  # CPU-only PyTorch
            self.assertEqual(e._sm120_route(torch.zeros(16, 8, dtype=torch.int32), torch.zeros(16, 8)), "stock")
        self.assertEqual(calls, [1])  # no vcap state / not capturing: stock


if __name__ == "__main__":
    unittest.main()
