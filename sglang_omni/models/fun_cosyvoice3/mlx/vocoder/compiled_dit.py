# SPDX-License-Identifier: Apache-2.0
"""Bound compiled DiT specializations without changing checkpoint parameters."""

from collections import OrderedDict
from threading import get_ident
from typing import Protocol

import mlx.core as mx
from mlx.utils import tree_flatten

from sglang_omni.models.fun_cosyvoice3.mlx.vocoder.dit import DiT

ArraySignature = tuple[tuple[int, ...], str] | None
ParameterLayout = tuple[tuple[str, tuple[int, ...], str], ...]
CompileSignature = tuple[tuple[ArraySignature, ...], int, str]


class RotaryEstimator(Protocol):
    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array | None,
        cond: mx.array | None,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array: ...


class CompiledDiT:
    def __init__(self, native: DiT, cache_size: int) -> None:
        if (
            not isinstance(cache_size, int)
            or isinstance(cache_size, bool)
            or cache_size <= 0
        ):
            raise ValueError("DiT compile cache size must be a positive integer")
        else:
            pass
        self.native: DiT = native
        self.cache_size: int = cache_size
        self.cache: OrderedDict[CompileSignature, RotaryEstimator] = OrderedDict()
        self.parameter_layout: ParameterLayout | None = None
        self.prepare()

    def prepare(self) -> None:
        """Check parameter layout once per request.

        Non-array module configuration changes require a new wrapper.
        """
        parameter_layout = tuple(
            (name, tuple(parameter.shape), str(parameter.dtype))
            for name, parameter in tree_flatten(self.native.parameters())
        )
        if parameter_layout != self.parameter_layout:
            self.cache.clear()
            self.parameter_layout = parameter_layout
        else:
            pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array,
        mu: mx.array,
        t: mx.array,
        spks: mx.array | None = None,
        cond: mx.array | None = None,
    ) -> mx.array:
        native = self.native
        cos, sin = native.rotary_embed.forward_from_seq_len(x.shape[-1])
        array_signatures = tuple(
            None if array is None else (tuple(array.shape), str(array.dtype))
            for array in (x, mask, mu, t, spks, cond, cos, sin)
        )
        signature: CompileSignature = (
            array_signatures,
            get_ident(),
            repr(mx.default_stream(mx.default_device())),
        )
        if signature in self.cache:
            compiled_body = self.cache[signature]
            self.cache.move_to_end(signature)
        else:

            def forward_with_rope(
                x: mx.array,
                mask: mx.array,
                mu: mx.array,
                t: mx.array,
                spks: mx.array | None,
                cond: mx.array | None,
                cos: mx.array,
                sin: mx.array,
            ) -> mx.array:
                return native.forward_with_rope(x, mask, mu, t, spks, cond, cos, sin)

            # note (Codex): Capture weights as inputs, and keep rotary cache evaluation outside tracing.
            compiled_body = mx.compile(
                forward_with_rope, inputs=native.state, shapeless=False
            )
            self.cache[signature] = compiled_body
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            else:
                pass
        return compiled_body(x, mask, mu, t, spks, cond, cos, sin)
