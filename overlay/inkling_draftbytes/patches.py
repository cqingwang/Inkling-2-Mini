"""The monkeypatches and the weight preparation they read.

Patch points (all draft-side; the target model and the accept path are untouched):

  sglang.srt.models.inkling2_mtp
    Inkling2ForConditionalGenerationMTP.attach_shared_modules  -> prepare weights after the stock attach
    Inkling2ForConditionalGenerationMTP.compute_base_logits    -> hotvocab
    Inkling2MTPHead.step_bias                                  -> fold8 / foldk
    Inkling2MTPHead.refresh_fold                               -> re-prepare in place on weight updates
    project_through_lm_head (the name inkling2_mtp imported)   -> gemv / head8
  sglang.srt.speculative.dspark_components.dspark_draft
    _draft_step                                                -> hotvocab
  draft layers' linear quant_method (instances, at prepare)    -> draftmx8
"""

from __future__ import annotations

import copy
import functools
import importlib
import json
import logging
import sys
from typing import Optional

import torch

from . import ops
from ._hooks import when_imported
from .quant import W8Pack, bf16_pack, quantize_mode

logger = logging.getLogger("inkling_draftbytes")

MTP_MODULE = "sglang.srt.models.inkling2_mtp"
DRAFT_MODULE = "sglang.srt.speculative.dspark_components.dspark_draft"
GEMV_MODULE = "sglang.kernels.ops.gemm.sm12x_bf16_gemv"


class DraftBytesPatchError(BaseException):
    """A patch target is missing or changed. Derives from BaseException on
    purpose: sglang's model registry swallows ``Exception`` raised while it
    imports model modules, which would turn a broken patch into a silent no-op."""


class _State:
    opts = None
    patched: list = []
    originals: dict = {}
    gemv_mod = None
    flag_override_noted = False


STATE = _State()


def say(msg: str) -> None:
    line = f"[inkling_draftbytes] {msg}"
    print(line, file=sys.stderr, flush=True)
    logger.info(msg)


def _fail(msg: str):
    say("FATAL: " + msg)
    raise DraftBytesPatchError(msg)


def _require(owner, names, where: str) -> None:
    missing = [n for n in names if not hasattr(owner, n)]
    if missing:
        _fail(f"patch target(s) {missing} not found in {where}; the image changed")


def _replace(owner, name: str, new, label: str) -> None:
    STATE.originals[label] = getattr(owner, name)
    setattr(owner, name, new)
    STATE.patched.append(label)


# --------------------------------------------------------------------------
# inkling2_mtp
# --------------------------------------------------------------------------


def patch_inkling2_mtp(mod) -> None:
    opts = STATE.opts
    _require(
        mod,
        ["Inkling2MTPHead", "Inkling2ForConditionalGenerationMTP", "project_through_lm_head"],
        MTP_MODULE,
    )
    head_cls = mod.Inkling2MTPHead
    model_cls = mod.Inkling2ForConditionalGenerationMTP
    _require(head_cls, ["step_bias", "refresh_fold", "_rnn_step", "attach_fold"], "Inkling2MTPHead")
    _require(model_cls, ["compute_base_logits", "attach_shared_modules"], "Inkling2ForConditionalGenerationMTP")

    orig_attach = model_cls.attach_shared_modules

    @functools.wraps(orig_attach)
    def attach_shared_modules(self, *args, **kwargs):
        result = orig_attach(self, *args, **kwargs)
        prepare(self)
        return result

    _replace(model_cls, "attach_shared_modules", attach_shared_modules,
             "Inkling2ForConditionalGenerationMTP.attach_shared_modules")

    orig_refresh = head_cls.refresh_fold

    @functools.wraps(orig_refresh)
    def refresh_fold(self, weight, multiplier):
        result = orig_refresh(self, weight, multiplier)
        model = getattr(self, "_idb_model", None)
        if model is not None:
            prepare(model, refresh=True)
        return result

    _replace(head_cls, "refresh_fold", refresh_fold, "Inkling2MTPHead.refresh_fold")

    if opts.gemv or opts.head8:
        orig_project = mod.project_through_lm_head
        flag_lookup = getattr(orig_project, "__globals__", {}).get("get_spec")
        quant_check = getattr(mod, "should_apply_lm_head_quant_method", None)

        def project_through_lm_head(hidden, lm_head):
            o = STATE.opts
            if quant_check is not None and quant_check(lm_head, lm_head.quant_method):
                return orig_project(hidden, lm_head)
            if flag_lookup is not None:
                try:
                    flag = getattr(flag_lookup(), "speculative_dspark_lm_head_dtype", None)
                except Exception:  # no runtime context (offline checks): flag unset
                    flag = None
                if flag is not None:
                    if not STATE.flag_override_noted:
                        STATE.flag_override_noted = True
                        say(f"--speculative-dspark-lm-head-dtype {flag} is set: "
                            "gemv/head8 stand down, the flag's path runs")
                    return orig_project(hidden, lm_head)
            pack = getattr(lm_head, "_idb_head", None)
            if pack is not None and hidden.shape[0] > 0:
                return ops.w8_linear(hidden, pack, None, lm_head.weight.dtype)
            if o.gemv and hidden.shape[0] > 0:
                return ops.bf16_gemv(hidden, lm_head.weight, o.gemv_max_m, STATE.gemv_mod)
            return orig_project(hidden, lm_head)

        _replace(mod, "project_through_lm_head", project_through_lm_head,
                 "inkling2_mtp.project_through_lm_head")

    if opts.fold_kernel:
        orig_step_bias = head_cls.step_bias

        @functools.wraps(orig_step_bias)
        def step_bias(self, prev_tokens, *, hidden_states, state):
            pack = getattr(self, "_idb_fold", None)
            if pack is None:
                return orig_step_bias(self, prev_tokens, hidden_states=hidden_states, state=state)
            state, f_rnn = state
            state, latent = self._rnn_step(state, prev_tokens, hidden_states, f_rnn)
            dtype = self._fold_weight.dtype
            bias = ops.w8_linear(latent.to(dtype), pack, self._fold_bias, dtype, narrow_k=True)
            if self.vocab_shard is None:
                bias = self._gather_logits(bias)
            return bias, (state, f_rnn)

        _replace(head_cls, "step_bias", step_bias, "Inkling2MTPHead.step_bias")

    if opts.hotvocab:
        orig_base = model_cls.compute_base_logits

        @functools.wraps(orig_base)
        def compute_base_logits(self, hidden):
            hot = getattr(self.markov_head, "_idb_hot", None)
            if hot is None:
                return orig_base(self, hidden)
            if self.logits_mup_width_multiplier:
                hidden = hidden / self.logits_mup_width_multiplier
            compact = hot_base_logits(hidden, hot)
            full = torch.full(
                (compact.shape[0], hot.width), float("-inf"),
                dtype=compact.dtype, device=compact.device,
            )
            full.index_copy_(1, hot.ids64, compact)
            return full, None

        _replace(model_cls, "compute_base_logits", compute_base_logits,
                 "Inkling2ForConditionalGenerationMTP.compute_base_logits")


def hot_base_logits(hidden: torch.Tensor, hot) -> torch.Tensor:
    opts = STATE.opts
    if hot.lm_head.scale_kind != 0:
        return ops.w8_linear(hidden, hot.lm_head, None, torch.bfloat16)
    weight = hot.lm_head.q
    if opts.gemv:
        return ops.bf16_gemv(hidden, weight, opts.gemv_max_m, STATE.gemv_mod)
    return torch.matmul(hidden.to(weight.dtype), weight.T)


# --------------------------------------------------------------------------
# dspark_draft
# --------------------------------------------------------------------------


def patch_dspark_draft(mod) -> None:
    _require(mod, ["_draft_step", "draft_markov_block", "MarkovSampleStep"], DRAFT_MODULE)
    orig_step = mod._draft_step

    @functools.wraps(orig_step)
    def _draft_step(markov_head, prev_tokens, base_logits, *, hidden_states, state,
                    temperatures, noise, step, logits_out, sharded_accept):
        hot = getattr(markov_head, "_idb_hot", None)
        if hot is None:
            return orig_step(
                markov_head, prev_tokens, base_logits, hidden_states=hidden_states,
                state=state, temperatures=temperatures, noise=noise, step=step,
                logits_out=logits_out, sharded_accept=sharded_accept,
            )
        state, f_rnn = state
        state, latent = markov_head._rnn_step(state, prev_tokens, hidden_states, f_rnn)
        step_dtype = markov_head.step_logits_dtype(base_logits.dtype)
        rows = base_logits.shape[0]
        # Full-width step logits: -inf off the hot set, so the kept logits are
        # the distribution the token is drawn from.
        full = torch.full((rows, hot.width), float("-inf"), dtype=step_dtype,
                          device=base_logits.device)
        ops.hot_step(
            latent, hot.fold, hot.fold_bias, base_logits, hot.ids32, full,
            softcap=markov_head.step_softcap, vocab_limit=markov_head.step_vocab_limit,
        )
        # Bias = an all-zero row (stride 0): x + 0 and the BF16 round are
        # identities, softcap and the hot roundings were applied above.
        tokens = mod.MarkovSampleStep.execute(
            base_logits=full,
            bias=hot.zero_row.expand(rows, -1),
            temperatures=temperatures,
            key=noise.key,
            offset=noise.offset,
            step=step,
            step_dtype=step_dtype,
            logits_out=logits_out,
            has_sampled_rows=None if logits_out is not None else False,
            softcap=None,
            vocab_limit=markov_head.step_vocab_limit,
        )
        return tokens, (state, f_rnn)

    _replace(mod, "_draft_step", _draft_step, "dspark_draft._draft_step")


# --------------------------------------------------------------------------
# weight preparation (runs once at draft attach, before any graph capture)
# --------------------------------------------------------------------------


class HotState:
    def __init__(self):
        self.ids64 = self.ids32 = None
        self.lm_head: Optional[W8Pack] = None
        self.fold: Optional[W8Pack] = None
        self.fold_bias = None
        self.zero_row = None
        self.width = 0
        self.n = 0

    def nbytes(self) -> int:
        size = self.lm_head.nbytes() + self.fold.nbytes()
        size += self.fold_bias.numel() * self.fold_bias.element_size()
        return size


def load_hot_ids(path: str, n: int, limit: int) -> list:
    """First n distinct ids in [0, limit) from a JSON list (frequency-ranked),
    returned sorted ascending."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("ids") or data.get("token_ids")
    if not isinstance(data, list):
        raise RuntimeError(f"{path}: expected a JSON list of token ids")
    seen, out = set(), []
    for value in data:
        tid = int(value)
        if 0 <= tid < limit and tid not in seen:
            seen.add(tid)
            out.append(tid)
            if len(out) == n:
                break
    if len(out) < n:
        say(f"hotvocab: {path} has only {len(out)} usable ids (< {limit}); using {len(out)}")
    if not out:
        raise RuntimeError(f"{path}: no usable token ids")
    return sorted(out)


def _requant(src, mode, old):
    if mode is None:
        return bf16_pack(src.contiguous())
    reuse = old if (old is not None and old.q.shape == src.shape and old.scale_kind != 0) else None
    return quantize_mode(src, mode, out=reuse)


def _build_hot(model, head, lm_head, old: Optional[HotState]) -> HotState:
    opts = STATE.opts
    width = int(lm_head.org_vocab_size)
    limit = min(width, int(head.step_vocab_limit or width))
    hot = old or HotState()
    device = lm_head.weight.device
    if old is None:
        ids = load_hot_ids(opts.hotvocab_file, opts.hotvocab, limit)
        hot.ids64 = torch.tensor(ids, dtype=torch.int64, device=device)
        hot.ids32 = hot.ids64.to(torch.int32)
        hot.width, hot.n = width, len(ids)
        hot.zero_row = torch.zeros(
            (1, width), dtype=head.step_logits_dtype(lm_head.weight.dtype), device=device
        )
    head_rows = lm_head.weight.index_select(0, hot.ids64)
    fold_rows = head._fold_weight.index_select(0, hot.ids64)
    if old is None:
        hot.lm_head = _requant(head_rows, opts.head8, None)
        hot.fold = _requant(fold_rows, opts.fold8, None)
        hot.fold_bias = head._fold_bias.index_select(0, hot.ids64)
    else:
        if old.lm_head.scale_kind == 0:
            old.lm_head.q.copy_(head_rows)
        else:
            _requant(head_rows, opts.head8, old.lm_head)
        if old.fold.scale_kind == 0:
            old.fold.q.copy_(fold_rows)
        else:
            _requant(fold_rows, opts.fold8, old.fold)
        old.fold_bias.copy_(head._fold_bias.index_select(0, hot.ids64))
    return hot


_W8_CLASSES: dict = {}


def _w8_method_class(cls):
    sub = _W8_CLASSES.get(cls)
    if sub is None:
        inner_apply = cls.apply

        def apply(self, layer, x, bias=None):
            pack = getattr(layer, "_idb_w8", None)
            if (
                pack is not None
                and x.is_cuda
                and x.dtype == torch.bfloat16
                and x.shape[-1] == pack.k
                and 0 < x.numel() // x.shape[-1] <= STATE.opts.w8_max_m
            ):
                return ops.w8_linear(x, pack, bias, x.dtype)
            return inner_apply(self, layer, x, bias)

        sub = type(f"DraftBytesW8{cls.__name__}", (cls,), {"apply": apply})
        _W8_CLASSES[cls] = sub
    return sub


def _draft_linears(model):
    found = []
    for name, module in model.layers.named_modules():
        method = getattr(module, "quant_method", None)
        weight = getattr(module, "weight", None)
        if (
            method is not None
            and "Unquantized" in type(method).__name__
            and isinstance(weight, torch.Tensor)
            and weight.dim() == 2
            and weight.dtype == torch.bfloat16
        ):
            found.append((name, module))
    return found


def _prepare_draftmx8(model, refresh: bool) -> int:
    mode = STATE.opts.draftmx8
    total = 0
    linears = _draft_linears(model)
    if not linears and not refresh:
        raise RuntimeError("draftmx8: no BF16 unquantized linears found under model.layers")
    for _, module in linears:
        old = getattr(module, "_idb_w8", None)
        module._idb_w8 = quantize_mode(module.weight.data, mode, out=old)
        if not refresh and not type(module.quant_method).__name__.startswith("DraftBytesW8"):
            method = copy.copy(module.quant_method)
            method.__class__ = _w8_method_class(type(module.quant_method))
            module.quant_method = method
        total += module._idb_w8.nbytes()
    return total


def _mb(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def prepare(model, refresh: bool = False) -> None:
    opts = STATE.opts
    head, lm_head = model.markov_head, model.lm_head
    if head.vocab_shard is not None or int(getattr(lm_head, "tp_size", 1)) != 1:
        raise RuntimeError("inkling_draftbytes supports TP1 only (vocab-sharded draft head found)")
    if lm_head.weight.dtype != torch.bfloat16:
        raise RuntimeError(f"inkling_draftbytes expects a BF16 LM head, got {lm_head.weight.dtype}")
    notes = []
    if not refresh:
        if opts.hotvocab:
            importlib.import_module(DRAFT_MODULE)
            if "dspark_draft._draft_step" not in STATE.patched:
                raise RuntimeError("hotvocab: dspark_draft._draft_step was not patched")
        if opts.gemv:
            gemv_mod = importlib.import_module(GEMV_MODULE)
            _require(gemv_mod, ["use_sm12x_bf16_gemv", "sm12x_bf16_gemv"], GEMV_MODULE)
            STATE.gemv_mod = gemv_mod
        head._idb_model = model
    if opts.fold_kernel:
        old = getattr(head, "_idb_fold", None)
        if opts.fold8:
            head._idb_fold = quantize_mode(head._fold_weight, opts.fold8, out=old)
            notes.append(f"fold8:{opts.fold8.label} {_mb(head._idb_fold.nbytes())}")
        else:
            head._idb_fold = bf16_pack(head._fold_weight)
            notes.append("foldk (BF16 fold, own kernel)")
    if opts.hotvocab:
        head._idb_hot = _build_hot(model, head, lm_head, getattr(head, "_idb_hot", None))
        hot = head._idb_hot
        notes.append(
            f"hotvocab {hot.n} ids of {hot.width} (head {hot.lm_head.label}, fold "
            f"{hot.fold.label}) {_mb(hot.nbytes())}"
        )
    elif opts.head8:
        lm_head._idb_head = quantize_mode(
            lm_head.weight.data, opts.head8, out=getattr(lm_head, "_idb_head", None)
        )
        notes.append(f"head8:{opts.head8.label} {_mb(lm_head._idb_head.nbytes())}")
    if opts.draftmx8:
        size = _prepare_draftmx8(model, refresh)
        notes.append(f"draftmx8:{opts.draftmx8.label} {len(_draft_linears(model))} linears {_mb(size)}")
    if opts.gemv:
        notes.append(f"gemv (sm12x <=16 rows, own kernel <= {opts.gemv_max_m})")
    say(
        ("refreshed" if refresh else "active")
        + f": options={opts.describe()}; patched {', '.join(STATE.patched)}; "
        + "; ".join(notes)
    )


# --------------------------------------------------------------------------


def arm(opts) -> None:
    STATE.opts = opts
    ops.configure(opts)
    when_imported(MTP_MODULE, patch_inkling2_mtp)
    if opts.hotvocab:
        when_imported(DRAFT_MODULE, patch_dspark_draft)
