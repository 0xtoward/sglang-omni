# SPDX-License-Identifier: Apache-2.0
"""A mixed step's decode tail on the GDN decode kernels.

A mixed step carries each running request as a one-row extend after the new prompts. SGLang's GDN backend runs the whole
batch through its prefill path: it copies every row's recurrent state out of the state pool and back and runs each one-row
tail through the chunked prefill kernel. The backend installed here cuts the tail off the batch and runs it the way a
decode step does, on the in-place conv update and recurrent decode kernels, as upstream SGLang does for deterministic
inference.
"""

from __future__ import annotations

import copy
import dataclasses
import types

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode


@dataclasses.dataclass(frozen=True, kw_only=True)
class DecodeTailSplit:
    """A mixed batch cut into its prompt rows and its one-row decode tail."""

    prompt_token_count: int
    tail_request_count: int
    prompt_batch: ForwardBatch
    prompt_metadata: object
    tail_batch: ForwardBatch
    tail_metadata: object


def decode_tail_request_count(forward_batch: ForwardBatch) -> int:
    """Running requests a mixed batch carries after its prompts: the trailing one-row extends with a prefix.

    Drama 3 prompts always start at prefix 0 (no radix cache, no prompt split), so a prefix marks a running request.
    """
    seq_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
        return 0
    elif seq_lens is None or prefix_lens is None:
        return 0
    else:
        count = 0
        for seq_len, prefix_len in zip(reversed(seq_lens), reversed(prefix_lens)):
            if seq_len == 1 and prefix_len > 0:
                count += 1
            else:
                break
        return count


def split_decode_tail(forward_batch: ForwardBatch, metadata) -> DecodeTailSplit | None:
    tail_request_count = decode_tail_request_count(forward_batch)
    seq_lens = forward_batch.extend_seq_lens_cpu
    prompt_request_count = 0 if seq_lens is None else len(seq_lens) - tail_request_count
    if tail_request_count == 0 or prompt_request_count <= 0:
        return None
    else:
        pass
    prompt_batch = copy.copy(forward_batch)
    prompt_batch.extend_seq_lens = forward_batch.extend_seq_lens[:prompt_request_count]
    prompt_batch.extend_prefix_lens = forward_batch.extend_prefix_lens[
        :prompt_request_count
    ]
    prompt_batch.extend_seq_lens_cpu = seq_lens[:prompt_request_count]
    prompt_batch.extend_prefix_lens_cpu = forward_batch.extend_prefix_lens_cpu[
        :prompt_request_count
    ]
    # note (0xtoward): rows are in request order and each one-row tail fills one conv block.
    block_table = metadata.conv_block_table
    prompt_metadata = dataclasses.replace(
        metadata,
        query_start_loc=metadata.query_start_loc[: prompt_request_count + 1],
        mamba_cache_indices=metadata.mamba_cache_indices[:prompt_request_count],
        conv_block_table=(
            None if block_table is None else block_table[:-tail_request_count]
        ),
    )
    tail_batch = copy.copy(forward_batch)
    tail_batch.forward_mode = ForwardMode.DECODE
    tail_batch.batch_size = tail_request_count
    tail_batch.mamba_track_mask = None
    tail_metadata = dataclasses.replace(
        metadata,
        query_start_loc=torch.arange(
            tail_request_count + 1,
            dtype=metadata.query_start_loc.dtype,
            device=metadata.query_start_loc.device,
        ),
        mamba_cache_indices=metadata.mamba_cache_indices[
            prompt_request_count : prompt_request_count + tail_request_count
        ],
        has_mamba_track_mask=False,
        mamba_track_indices=None,
    )
    return DecodeTailSplit(
        prompt_token_count=sum(seq_lens[:prompt_request_count]),
        tail_request_count=tail_request_count,
        prompt_batch=prompt_batch,
        prompt_metadata=prompt_metadata,
        tail_batch=tail_batch,
        tail_metadata=tail_metadata,
    )


def narrow_rows(tensor: torch.Tensor, start: int, length: int) -> torch.Tensor:
    return tensor.narrow(0 if tensor.ndim == 2 else 1, start, length)


def install_gdn_decode_tail(attention_backend: HybridLinearAttnBackend) -> None:
    """Make the engine's GDN backend run every mixed batch's decode tail on the decode kernels."""
    backend = attention_backend.linear_attn_backend
    if not isinstance(backend, GDNAttnBackend):
        raise TypeError(
            f"the Drama 3 decode tail split needs the GDN linear attention backend; got "
            f"{type(backend).__name__}"
        )
    else:
        pass
    backend.decode_tail_split = None
    original_init = backend.init_forward_metadata
    original_out_graph_init = backend.init_forward_metadata_out_graph
    original_forward_extend = backend.forward_extend

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        original_init(forward_batch)
        self.decode_tail_split = (
            split_decode_tail(forward_batch, self.forward_metadata)
            if self.mis_metadata is None
            else None
        )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ) -> None:
        original_out_graph_init(forward_batch, in_capture=in_capture)
        self.decode_tail_split = None

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        split = self.decode_tail_split
        if (
            split is None
            or not isinstance(mixed_qkv, torch.Tensor)
            or mixed_qkv.shape[0] != split.prompt_token_count + split.tail_request_count
        ):
            return original_forward_extend(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )
        else:
            pass
        kwargs.pop("linear_attn_output", None)
        prompt_tokens = split.prompt_token_count
        tail_requests = split.tail_request_count
        metadata = self.forward_metadata
        try:
            self.forward_metadata = split.prompt_metadata
            prompt_output = original_forward_extend(
                layer,
                split.prompt_batch,
                mixed_qkv[:prompt_tokens],
                narrow_rows(a, 0, prompt_tokens),
                narrow_rows(b, 0, prompt_tokens),
                **kwargs,
            )
            self.forward_metadata = split.tail_metadata
            tail_output = self.forward_decode(
                layer,
                split.tail_batch,
                mixed_qkv[prompt_tokens:],
                narrow_rows(a, prompt_tokens, tail_requests),
                narrow_rows(b, prompt_tokens, tail_requests),
                **kwargs,
            )
        finally:
            self.forward_metadata = metadata
        tail_output = tail_output.reshape(
            *prompt_output.shape[:-3], tail_requests, *prompt_output.shape[-2:]
        )
        return torch.cat([prompt_output, tail_output], dim=-3)

    backend.init_forward_metadata = types.MethodType(init_forward_metadata, backend)
    backend.init_forward_metadata_out_graph = types.MethodType(
        init_forward_metadata_out_graph, backend
    )
    backend.forward_extend = types.MethodType(forward_extend, backend)


__all__ = ["install_gdn_decode_tail", "split_decode_tail", "decode_tail_request_count"]
