"""The import-time patches. Each hook validates its targets (presence, signature,
and a few source-level invariants) and raises on any mismatch, so a changed engine
fails at startup instead of silently running without the cap."""

from __future__ import annotations

import functools
import importlib
import inspect
from typing import Callable, Iterable

import torch

from inkling_vcap.state import PREFIX, STATE, log

MOD_MOE = "sglang.srt.models.inkling_common.inkling2_moe"
MOD_VERIFY = "sglang.srt.speculative.dspark_components.dspark_verify"
MOD_SAG = "sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph"
MOD_DRAFT = "sglang.srt.speculative.dspark_components.dspark_draft"
MOD_WORKER = "sglang.srt.speculative.dspark_components.dspark_worker_v2"

P_MOE = "Inkling2MoE.forward"
P_GATE = "Inkling2Gate.forward"
P_STATIC_ACCEPT = "DsparkVerifyEpilogue._accept_static_greedy"
P_RUN_NON_COMPACT = "TargetVerifyExecutor.run_non_compact"
P_ACCEPT_VERIFY = "dspark_verify.accept_draft_tokens"
P_ACCEPT_SAG = "dspark_sampling_accept_graph.accept_draft_tokens"
P_PROPOSE = "DraftBlockProposer.propose"
P_WORKER_INIT = "DSparkWorkerV2.__init__"
P_FORWARD_DECODE = "DSparkWorkerV2._forward_decode"

REQUIRED = (
    P_MOE, P_GATE, P_STATIC_ACCEPT, P_RUN_NON_COMPACT, P_ACCEPT_VERIFY,
    P_ACCEPT_SAG, P_PROPOSE, P_WORKER_INIT, P_FORWARD_DECODE,
)
_MARK = "__inkling_vcap__"


class PatchTargetError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers
def _attr(owner, name: str, where: str):
    value = getattr(owner, name, None)
    if value is None:
        raise PatchTargetError(f"{PREFIX} patch target missing: {where}.{name}")
    return value


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def _require_params(fn, expected: list[str], where: str) -> None:
    got = _params(fn)
    if got != expected:
        raise PatchTargetError(
            f"{PREFIX} signature changed for {where}: expected {expected}, got {got}"
        )


def _require_subset(fn, names: Iterable[str], where: str) -> None:
    got = set(_params(fn))
    missing = [n for n in names if n not in got]
    if missing:
        raise PatchTargetError(
            f"{PREFIX} signature changed for {where}: missing parameters {missing}"
        )


def _require_source(fn, needles: list[str], where: str, ordered: bool = False) -> None:
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as exc:
        raise PatchTargetError(f"{PREFIX} cannot read the source of {where}") from exc
    last = -1
    for needle in needles:
        idx = src.find(needle)
        if idx < 0:
            raise PatchTargetError(
                f"{PREFIX} {where} no longer contains {needle!r}; re-check the patch"
            )
        if ordered and idx < last:
            raise PatchTargetError(f"{PREFIX} {where}: unexpected order of {needles}")
        last = idx


def _mark(fn: Callable, original: Callable) -> Callable:
    functools.update_wrapper(fn, original)
    setattr(fn, _MARK, True)
    return fn


def _is_marked(fn) -> bool:
    return bool(getattr(fn, _MARK, False))


# ------------------------------------------------------------------ MoE
def patch_moe(mod) -> None:
    moe_cls = _attr(mod, "Inkling2MoE", MOD_MOE)
    gate_cls = _attr(mod, "Inkling2Gate", MOD_MOE)
    if _is_marked(moe_cls.forward) and _is_marked(gate_cls.forward):
        STATE.applied.update((P_MOE, P_GATE))
        return
    _require_params(moe_cls.forward, ["self", "x", "forward_batch", "combine"], P_MOE)
    _require_params(gate_cls.forward, ["self", "x"], P_GATE)
    _require_source(moe_cls.forward, ["self.gate(x)", "self.experts("], P_MOE, ordered=True)

    orig_moe = moe_cls.forward
    orig_gate = gate_cls.forward

    def moe_forward(self, x, forward_batch=None, combine=True):
        prev = STATE.in_verify_moe
        mode = getattr(forward_batch, "forward_mode", None)
        STATE.in_verify_moe = bool(
            STATE.enabled and mode is not None and mode.is_target_verify()
        )
        try:
            return orig_moe(self, x, forward_batch, combine)
        finally:
            STATE.in_verify_moe = prev

    def gate_forward(self, x):
        out = orig_gate(self, x)
        if not (STATE.enabled and STATE.in_verify_moe):
            return out
        weights, ids = out
        cfg = STATE.config
        if STATE.capping and cfg.remap:
            weights, ids = STATE.remap(weights, ids)
        if cfg.union:
            # Observational only: never changes what the experts receive.
            STATE.count_union(self, ids, int(self.weight.shape[0]))
        return weights, ids

    moe_cls.forward = _mark(moe_forward, orig_moe)
    gate_cls.forward = _mark(gate_forward, orig_gate)
    STATE.applied.update((P_MOE, P_GATE))


# ------------------------------------------------------------------ verify / accept
def _wrap_accept_draft_tokens(orig):
    _require_subset(orig, ["candidates", "cutoff_layout"], "accept_draft_tokens")
    sig = inspect.signature(orig)
    if sig.parameters["candidates"].kind is not inspect.Parameter.KEYWORD_ONLY:
        raise PatchTargetError(f"{PREFIX} accept_draft_tokens is no longer keyword-only")

    def accept_draft_tokens(**kwargs):
        if STATE.enabled and kwargs.get("cutoff_layout") is None:
            cutoff = STATE.cutoff_for(int(kwargs["candidates"].shape[0]))
            if cutoff is not None:
                kwargs["cutoff_layout"] = cutoff
        return orig(**kwargs)

    return _mark(accept_draft_tokens, orig)


def patch_verify(mod) -> None:
    epi = _attr(mod, "DsparkVerifyEpilogue", MOD_VERIFY)
    execu = _attr(mod, "TargetVerifyExecutor", MOD_VERIFY)
    adt = _attr(mod, "accept_draft_tokens", MOD_VERIFY)

    if not _is_marked(epi._accept_static_greedy):
        _require_params(
            epi._accept_static_greedy,
            ["self", "io", "verify_lens", "gate", "gated"],
            P_STATIC_ACCEPT,
        )
        _require_subset(epi._accept, ["candidates", "logits", "draft_tokens",
                                      "seq_lens", "cutoff_verify_lens"],
                        "DsparkVerifyEpilogue._accept")
        _require_source(epi._static_epilogue, ["self._accept_static_greedy"],
                        "DsparkVerifyEpilogue._static_epilogue")
        _require_source(epi.run_static, ["verify_lens = self.verify_lens_buf[:bs]",
                                         "accept(io, verify_lens=verify_lens"],
                        "DsparkVerifyEpilogue.run_static")
        orig_static = epi._accept_static_greedy

        def _accept_static_greedy(self, io, *, verify_lens, gate, gated):
            if not (STATE.enabled and STATE.capping):
                return orig_static(self, io, verify_lens=verify_lens, gate=gate,
                                   gated=gated)
            # The greedy accept with the cutoff, recorded into the verify graph; the
            # fused accept tail (which never caps) is bypassed while the cap is on.
            STATE.static_accept_recorded += 1
            commit_lens = self._accept(
                candidates=io.candidates,
                logits=io.logits,
                draft_tokens=io.candidates[:, 1:].contiguous(),
                seq_lens=io.seq_lens,
                cutoff_verify_lens=verify_lens,
            )
            if not gated:
                return None
            return torch.minimum(commit_lens, verify_lens.to(torch.int32)) * gate

        epi._accept_static_greedy = _mark(_accept_static_greedy, orig_static)

    if not _is_marked(execu.run_non_compact):
        _require_params(
            execu.run_non_compact,
            ["self", "batch", "draft_input", "verify_ids_2d", "verify_window",
             "sampling_info"],
            P_RUN_NON_COMPACT,
        )
        orig_rnc = execu.run_non_compact

        def run_non_compact(self, *args, **kwargs):
            if args:
                raise PatchTargetError(
                    f"{PREFIX} run_non_compact called positionally; re-check the patch"
                )
            if STATE.enabled:
                _before_verify(int(kwargs["verify_ids_2d"].shape[0]))
            return orig_rnc(self, **kwargs)

        execu.run_non_compact = _mark(run_non_compact, orig_rnc)

    if not _is_marked(adt):
        mod.accept_draft_tokens = _wrap_accept_draft_tokens(adt)
    STATE.applied.update((P_STATIC_ACCEPT, P_RUN_NON_COMPACT, P_ACCEPT_VERIFY))


def patch_sampling_accept_graph(mod) -> None:
    adt = _attr(mod, "accept_draft_tokens", MOD_SAG)
    _require_source(_attr(mod, "sampling_accept_commit_lens", MOD_SAG),
                    ["accept_draft_tokens("], "sampling_accept_commit_lens")
    if not _is_marked(adt):
        mod.accept_draft_tokens = _wrap_accept_draft_tokens(adt)
    STATE.applied.add(P_ACCEPT_SAG)


def _before_verify(bs: int) -> None:
    if not STATE._validated_first_step:
        _first_step_checks()
        STATE._validated_first_step = True
    proposal, STATE.last_proposal = STATE.last_proposal, None
    if proposal is None and STATE.config.uses_draft_probs:
        raise RuntimeError(f"{PREFIX} target verify without a stashed draft proposal")
    STATE.plan_step(bs=bs, proposal=proposal)


def _first_step_checks() -> None:
    worker = STATE.worker
    cfg = STATE.config
    sampler = getattr(worker, "_draft_sampler", None)
    if cfg.uses_draft_probs and sampler is not None:
        strategy = type(getattr(sampler, "_draft_strategy", None)).__name__
        if strategy != "DsparkDraftStrategy" or getattr(sampler, "_markov_walker", None):
            raise RuntimeError(
                f"{PREFIX} {cfg.kind} mode reads every row of the draft step logits, "
                f"but the draft sampler runs {strategy} / int8 markov walk, which may "
                "leave greedy rows unwritten. Use fixed:K, or disable that path."
            )
    epilogue = getattr(worker, "_verify_epilogue", None)
    captured = bool(getattr(epilogue, "static_io", None))
    if (
        STATE.capping and epilogue is not None and captured and cfg.remap
        and STATE.remap_recorded > 0 and STATE.static_accept_recorded == 0
    ):
        raise RuntimeError(
            f"{PREFIX} remap was recorded in the verify graphs but the in-graph accept "
            "cap was not; refusing to run (output would not be exact)."
        )
    if not STATE.capping:
        log("full width for every batch size (e.g. fixed:8): verification is "
            "untouched (no remap, no cutoff, stock accept tail); stats/union only")
    elif cfg.remap and STATE.remap_recorded == 0:
        log("WARNING: no MoE remap recorded before the first decode step "
            "(verify graphs captured before the overlay attached?); the cap stays "
            "exact but saves nothing.")
    log(
        f"first decode step: remap recorded in {STATE.remap_recorded} MoE calls, "
        f"in-graph capped accept recorded {STATE.static_accept_recorded}x, "
        f"captured static verify bs={sorted(getattr(epilogue, 'static_io', {}) or [])}"
    )
    if STATE._union_acc is not None:
        STATE._union_acc.zero_()  # drop warmup / capture counts


# ------------------------------------------------------------------ draft
def patch_draft(mod) -> None:
    proposer = _attr(mod, "DraftBlockProposer", MOD_DRAFT)
    if not _is_marked(proposer.propose):
        _require_subset(proposer.propose, ["batch", "draft_input", "bs",
                                           "sampling_info"], P_PROPOSE)
        orig = proposer.propose

        def propose(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            if STATE.enabled:
                STATE.last_proposal = out
            return out

        proposer.propose = _mark(propose, orig)
    STATE.applied.add(P_PROPOSE)


# ------------------------------------------------------------------ worker
def patch_worker(mod) -> None:
    worker_cls = _attr(mod, "DSparkWorkerV2", MOD_WORKER)
    _require_source(
        worker_cls._forward_decode,
        ["self._proposer.propose(", "begin_static_step(", "run_non_compact(",
         "accept_and_finalize("],
        P_FORWARD_DECODE,
        ordered=True,
    )
    if not _is_marked(worker_cls.__init__):
        orig_init = worker_cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if getattr(self, "_hosts_draft", False):
                attach(self)

        worker_cls.__init__ = _mark(__init__, orig_init)
    if not _is_marked(worker_cls._forward_decode):
        orig_fd = worker_cls._forward_decode

        def _forward_decode(self, *args, **kwargs):
            result = orig_fd(self, *args, **kwargs)
            if STATE.enabled and STATE.worker is self:
                STATE.observe(result)
            return result

        worker_cls._forward_decode = _mark(_forward_decode, orig_fd)
    STATE.applied.update((P_WORKER_INIT, P_FORWARD_DECODE))


HOOKS = {
    MOD_MOE: patch_moe,
    MOD_VERIFY: patch_verify,
    MOD_SAG: patch_sampling_accept_graph,
    MOD_DRAFT: patch_draft,
    MOD_WORKER: patch_worker,
}


def _runtime_context():
    try:
        return importlib.import_module("sglang.srt.runtime_context")
    except ImportError:
        return None


def _capacity_without_epilogue() -> int:
    rc = _runtime_context()
    sizes = [1]
    if rc is not None:
        try:
            sizes.append(max(rc.get_exec().graph.cuda_graph_config.decode.bs))
        except Exception:
            pass
        try:
            sizes.append(int(rc.get_schedule().max_running_requests or 0))
        except Exception:
            pass
    return max(sizes)


def attach(worker) -> None:
    """Validate the serving setup, allocate the buffers, and enable the hooks."""
    cfg = STATE.config
    for name in HOOKS:
        importlib.import_module(name)  # triggers the hook if not yet imported
    missing = [p for p in REQUIRED if p not in STATE.applied]
    if missing:
        raise PatchTargetError(f"{PREFIX} patches not applied: {missing}")

    stride = int(worker.verify_num_draft_tokens)
    cfg.validate_for_stride(stride)
    planner = getattr(worker, "_verify_planner", None)
    if getattr(planner, "mode_value", None) != "static":
        raise RuntimeError(
            f"{PREFIX} needs SGLANG_RAGGED_VERIFY_MODE=static (got "
            f"{getattr(planner, 'mode_value', None)!r}); the cap replaces ragged verify"
        )
    if float(getattr(worker, "_simulate_acc_len", 0.0) or 0.0) > 0:
        raise RuntimeError(f"{PREFIX} incompatible with SGLANG_SIMULATE_ACC_LEN")
    rc = _runtime_context()
    if rc is not None:
        spec = rc.get_spec()
        if getattr(spec, "speculative_use_block_verification", False):
            raise RuntimeError(
                f"{PREFIX} incompatible with --speculative-use-block-verification"
            )
        par = rc.get_parallel()

        def _p(name, default):
            try:
                return getattr(par, name)
            except Exception:
                return default

        if int(_p("pp_size", 1) or 1) > 1:
            raise RuntimeError(f"{PREFIX} pipeline parallelism is not supported")
        if cfg.uses_draft_probs and (
            int(_p("tp_size", 1) or 1) > 1 or bool(_p("attn_dp_enabled", False))
        ):
            raise RuntimeError(
                f"{PREFIX} {cfg.kind} mode is validated for tp=1 only (rank-invariant "
                "live lengths are not synced); use fixed:K with tp>1"
            )
    if cfg.uses_draft_probs and getattr(worker, "_greedy_draft", False):
        raise RuntimeError(
            f"{PREFIX} {cfg.kind} mode needs draft step logits; "
            "SGLANG_DSPARK_USE_GREEDY_DRAFT=1 drops them (use fixed:K)"
        )
    if getattr(worker, "_sharded_accept", False):
        raise RuntimeError(f"{PREFIX} the vocab-sharded accept is not supported")

    moe_mod = importlib.import_module(MOD_MOE)
    target = getattr(worker, "target_worker", None) or worker._target_worker
    model = target.model_runner.model
    n_moe = 0
    for name, module in model.named_modules():
        if isinstance(module, moe_mod.Inkling2MoE):
            short = name.replace("model.", "").replace("layers.", "L").replace(".mlp", "")
            module._vcap_name = short
            module.gate._vcap_name = short
            n_moe += 1
    if n_moe == 0:
        raise RuntimeError(
            f"{PREFIX} the target model has no Inkling2MoE layers; the overlay only "
            "supports Inkling2 targets"
        )

    epilogue = getattr(worker, "_verify_epilogue", None)
    if epilogue is not None:
        if getattr(epilogue, "simulate_fold", False):
            raise RuntimeError(f"{PREFIX} incompatible with SGLANG_DSPARK_SIMULATE_FOLD")
        if int(epilogue.stride) != stride:
            raise RuntimeError(f"{PREFIX} epilogue stride {epilogue.stride} != {stride}")
        capacity = int(epilogue.max_bs)
        STATE.allocate(stride=stride, capacity=capacity, device=worker.device,
                       live_lens=epilogue.verify_lens_buf)
        STATE.live_lens_source = "DsparkVerifyEpilogue.verify_lens_buf"
    else:
        capacity = max(_capacity_without_epilogue(), 256)
        STATE.allocate(stride=stride, capacity=capacity, device=worker.device)
        STATE.live_lens_source = "own buffer (no static epilogue: eager accept)"
    STATE.epilogue = epilogue
    STATE.capping = not cfg.never_caps(stride)
    fused_note = "off"
    if STATE.capping and cfg.remap and cfg.fused:
        from inkling_vcap import fused

        if torch.device(worker.device).type != "cuda":
            fused_note = "off (not CUDA)"
        else:
            err = fused.selfcheck(worker.device)
            STATE.use_fused = err is None
            fused_note = "on (self-check passed)" if err is None else f"off ({err})"
    STATE.worker = worker
    STATE.enabled = True
    log(
        f"enabled mode={cfg.describe()} capping={'on' if STATE.capping else 'off (inert)'} "
        f"stride={stride} capacity={capacity} reqs "
        f"moe_layers={n_moe} live_lens={STATE.live_lens_source} "
        f"fused_remap={fused_note}; patched: "
        + ", ".join(REQUIRED)
    )
