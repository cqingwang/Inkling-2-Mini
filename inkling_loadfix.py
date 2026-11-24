#!/usr/bin/env python3
"""GB10 safetensors loader compatibility shim for Inkling-2-Mini.

Directly copying a tensor backed by a safetensors mmap into GB10 unified memory
can stall for minutes or indefinitely inside cuMemcpyHtoDAsync while DeepSeek
holds most of the sibling unified-memory pool. The same bytes copied through an
anonymous CPU tensor complete in milliseconds. Materialize one checkpoint tensor
at a time before the normal SGLang weight loaders see it, then enter the normal
SGLang server.
"""

from __future__ import annotations

import runpy


def install() -> None:
    from sglang.srt.model_loader import loader as model_loader

    iterator_names = (
        "safetensors_weights_iterator",
        "buffered_multi_thread_safetensors_weights_iterator",
    )
    for name in iterator_names:
        original = getattr(model_loader, name, None)
        if original is None or getattr(original, "_inkling_mmap_clone_fix", False):
            continue

        def clone_iterator(*args, _original=original, **kwargs):
            for weight_name, tensor in _original(*args, **kwargs):
                yield weight_name, tensor.clone()

        clone_iterator._inkling_mmap_clone_fix = True
        setattr(model_loader, name, clone_iterator)


if __name__ == "__main__":
    install()
    runpy.run_module("sglang.launch_server", run_name="__main__")
