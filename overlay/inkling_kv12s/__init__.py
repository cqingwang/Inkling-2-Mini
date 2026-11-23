"""Packed-only Inkling global KV overlay."""

import os
import sys


def enabled(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "none", "false", "no")


class IncompatibleImage(BaseException):
    """Startup mismatch that must propagate through the model registry."""


_installed = False


def install():
    global _installed
    if not enabled("INKLING_KV12S"):
        return
    if enabled("INKLING_KV12"):
        raise IncompatibleImage("INKLING_KV12S and INKLING_KV12 are mutually exclusive")
    if _installed:
        return
    from inkling_attn import install as install_attn, parse_levers
    from .patch import register

    levers = parse_levers(os.environ.get("INKLING_ATTN", ""))
    install_attn(tuple(dict.fromkeys((*levers, "prefill_global"))))
    register()
    _installed = True
    print(f"[inkling_kv12s pid={os.getpid()}] enabled packed-only global KV", file=sys.stderr, flush=True)
