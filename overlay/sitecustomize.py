"""Combined loader for the Inkling-2-Mini SGLang overlays (verify cap, draft bytes, kernels, pico12, prefill).
Each overlay activates only when its environment switch is set; with none set nothing is imported.
A failing overlay stops the process (site.py would otherwise only print a warning and run the stock path)."""
import os
import sys


def _on(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "none", "false", "no")


try:
    if _on("INKLING_VERIFY_CAP"):
        from inkling_vcap import activate
        activate()
    if _on("INKLING_DRAFT_BYTES"):
        import inkling_draftbytes
        inkling_draftbytes.install()
    if _on("INKLING_KERNELS"):
        import inkling_kernels
        inkling_kernels.install()
    if _on("INKLING_PICO12"):  # after inkling_kernels: its _project_sources wrapper goes outside qkvr_gemv's
        import inkling_pico12
        inkling_pico12.install()
    if _on("INKLING_PREFILL"):  # last: its _project_sources wrapper (large M only) goes outermost
        import inkling_prefill
        inkling_prefill.install()
except BaseException as exc:  # noqa: BLE001
    sys.stderr.write("[inkling-overlay pid=%d] FATAL at startup: %r\n" % (os.getpid(), exc))
    sys.stderr.flush()
    os._exit(78)
