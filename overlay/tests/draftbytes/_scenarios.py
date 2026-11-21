"""Scenarios run in a fresh interpreter (one per test) against the CPU stubs.
Usage: python _scenarios.py NAME  (prints one JSON line)."""

import json
import sys

import torch


def _setup(seed=0, padded=4096, org=4000, limit=3990, softcap=30.0):
    torch.manual_seed(seed)
    import inkling_draftbytes

    opts = inkling_draftbytes.install()
    from sglang.srt.models import inkling2_mtp as m

    model = m.Inkling2ForConditionalGenerationMTP(padded=padded, org=org, limit=limit, softcap=softcap)
    lm_head = m.LMHead(padded, org, 128)
    model.attach_shared_modules(embed_tokens=None, lm_head=lm_head)
    return opts, m, model


def _block(model, dd, bs=2, gamma=7, temps=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(bs * gamma, 128, generator=g) * 3).to(torch.bfloat16)
    draft_hidden = torch.randn(bs, gamma, 128, generator=g).to(torch.bfloat16)
    model.markov_head._f_rnn = torch.randn(bs, 128, generator=g).to(torch.bfloat16)
    anchor = torch.randint(0, 3000, (bs,), generator=g)
    base, _ = model.compute_base_logits(hidden)
    base = base.view(bs, gamma, -1)
    temps = torch.zeros(bs) if temps is None else temps
    noise = dd.DraftNoise(key=123, offset=torch.zeros(1, dtype=torch.int64))
    tokens, logits = dd.draft_markov_block(
        markov_head=model.markov_head, base_logits=base, anchor_tokens=anchor,
        draft_hidden=draft_hidden, step_temperatures=temps, noise=noise,
    )
    return tokens, logits


def noop():
    import sys as _s

    n_before = len(_s.meta_path)
    import inkling_draftbytes

    res = inkling_draftbytes.install()
    from sglang.srt.models import inkling2_mtp as m

    return {
        "installed": res is not None,
        "meta_path_grew": len(_s.meta_path) != n_before,
        "step_bias_stock": m.Inkling2MTPHead.step_bias.__module__ == m.__name__,
        "project_stock": m.project_through_lm_head.__module__ == m.__name__,
    }


def fold_compare():
    """Patched step_bias (fold8 or foldk per env) vs the stock step_bias."""
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    head = model.markov_head
    stock = patches.STATE.originals["Inkling2MTPHead.step_bias"]
    bs = 4
    g = torch.Generator().manual_seed(5)
    state = (torch.zeros(bs, 64, dtype=torch.bfloat16),
             torch.randn(bs, 128, generator=g).to(torch.bfloat16))
    prev = torch.randint(0, 3000, (bs,), generator=g)
    hid = torch.randn(bs, 128, generator=g).to(torch.bfloat16)
    ref, _ = stock(head, prev, hidden_states=hid, state=state)
    new, _ = head.step_bias(prev, hidden_states=hid, state=state)
    diff = (new.float() - ref.float())
    rel = (diff.norm() / ref.float().norm()).item()
    ulp1 = (diff.abs() <= ref.float().abs() * 2 ** -7 + 1e-30).float().mean().item()
    return {"shape_ok": list(new.shape) == list(ref.shape), "dtype": str(new.dtype),
            "rel_err": rel, "within_1ulp": ulp1, "patched": patches.STATE.patched}


def hot_compare():
    """hotvocab (+ any other env options) vs stock, through the draft loop."""
    from inkling_draftbytes import patches
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    opts, m, model = _setup()
    hot = model.markov_head._idb_hot
    ids = hot.ids64
    tokens, logits = _block(model, dd)
    # The stock run on the same model: drop the hot state, restore stock paths.
    model.markov_head._idb_hot = None
    model.markov_head._idb_fold = None
    model.lm_head._idb_head = None
    tokens_ref, logits_ref = _block(model, dd)
    in_hot = torch.zeros(logits.shape[-1], dtype=torch.bool)
    in_hot[ids] = True
    off = logits[..., ~in_hot]
    # Compare only steps whose earlier drafted tokens agree (same recurrent input).
    same_prefix = torch.ones_like(tokens, dtype=torch.bool)
    same_prefix[:, 1:] = (tokens == tokens_ref).cumprod(1)[:, :-1].bool()
    on = logits[..., in_hot][same_prefix].float()
    on_ref = logits_ref[..., in_hot][same_prefix].float()
    finite = torch.isfinite(on_ref)
    d = (on - on_ref)[finite].abs()
    tol = on_ref[finite].abs() * 2 ** -7
    ref_in_hot = in_hot[tokens_ref] & same_prefix
    return {
        "n_hot": int(ids.numel()),
        "compared_steps": int(same_prefix.sum()),
        "off_all_neginf": bool(torch.isneginf(off.float()).all()),
        "hot_exact_frac": (d == 0).float().mean().item(),
        "hot_within_1ulp": (d <= tol).float().mean().item(),
        "hot_neginf_match": bool((torch.isneginf(on) == torch.isneginf(on_ref)).all()),
        "tokens_all_hot": bool(in_hot[tokens].all()),
        "agree_when_ref_hot": (tokens[ref_in_hot] == tokens_ref[ref_in_hot]).float().mean().item(),
        "patched": patches.STATE.patched,
    }


def hot_sampled():
    """Sampled rows: the kept logits must be what the token was drawn from:
    every drawn token is hot and has a finite kept logit."""
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    opts, m, model = _setup()
    ids = model.markov_head._idb_hot.ids64
    in_hot = torch.zeros(4000, dtype=torch.bool)
    in_hot[ids] = True
    ok = True
    for seed in range(5):
        tokens, logits = _block(model, dd, bs=3, temps=torch.tensor([0.7, 1.0, 0.0]), seed=seed)
        kept = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ok &= bool(in_hot[tokens].all()) and bool(torch.isfinite(kept.float()).all())
    return {"ok": ok}


def head_compare():
    """gemv/head8 via project_through_lm_head vs stock matmul."""
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    stock = patches.STATE.originals["inkling2_mtp.project_through_lm_head"]
    hidden = (torch.randn(7, 128) * 3).to(torch.bfloat16)
    ref = stock(hidden, model.lm_head).float()
    new = m.project_through_lm_head(hidden, model.lm_head).float()
    rel = ((new - ref).norm() / ref.norm()).item()
    return {"rel_err": rel, "argmax_agree": (new.argmax(-1) == ref.argmax(-1)).float().mean().item()}


def draftmx8():
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    lin = model.layers[0].qkv_proj
    x = torch.randn(8, 128).to(torch.bfloat16)
    out, _ = lin(x)  # CPU: falls back to the inner (stock) apply
    ref = torch.nn.functional.linear(x, lin.weight)
    pack = lin._idb_w8
    from inkling_draftbytes.quant import linear_ref

    q = linear_ref(x, pack).float()
    rel = ((q - ref.float()).norm() / ref.float().norm()).item()
    return {
        "wrapped": type(lin.quant_method).__name__,
        "isinstance_kept": isinstance(lin.quant_method, m.UnquantizedLinearMethod),
        "cpu_fallback_exact": bool(torch.equal(out, ref)),
        "pack_label": pack.label,
        "w8_rel_err": rel,
        "n_wrapped": sum(1 for l in model.layers for p in (l.qkv_proj, l.o_proj)
                         if type(p.quant_method).__name__.startswith("DraftBytesW8")),
    }


def import_only():
    from sglang.srt.models import inkling2_mtp  # noqa: F401

    return {"imported": True}


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]]()))
