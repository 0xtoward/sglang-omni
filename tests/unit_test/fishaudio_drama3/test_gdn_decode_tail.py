# SPDX-License-Identifier: Apache-2.0
"""A mixed batch's decode tail is cut off for the GDN decode kernels."""

from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.models.fishaudio_drama3.gdn_decode_tail import (
    decode_tail_request_count,
    split_decode_tail,
)


def mixed_batch(
    seq_lens: list[int], prefix_lens: list[int], forward_mode: ForwardMode
) -> SimpleNamespace:
    return SimpleNamespace(
        forward_mode=forward_mode,
        batch_size=len(seq_lens),
        extend_seq_lens=torch.tensor(seq_lens),
        extend_prefix_lens=torch.tensor(prefix_lens),
        extend_seq_lens_cpu=seq_lens,
        extend_prefix_lens_cpu=prefix_lens,
        mamba_track_mask=torch.tensor([True] * len(seq_lens)),
    )


def test_a_mixed_batch_splits_into_its_prompts_and_its_one_row_tails() -> None:
    batch = mixed_batch([5, 3, 1, 1, 1], [0, 0, 10, 20, 30], ForwardMode.MIXED)
    metadata = ForwardMetadata(
        query_start_loc=torch.tensor([0, 5, 8, 9, 10, 11]),
        mamba_cache_indices=torch.tensor([7, 8, 9, 10, 11]),
        conv_block_table=torch.arange(6).unsqueeze(-1),
    )

    split = split_decode_tail(batch, metadata)

    assert split is not None
    assert (split.prompt_token_count, split.tail_request_count) == (8, 3)
    assert split.prompt_batch.extend_seq_lens_cpu == [5, 3]
    assert split.prompt_batch.extend_prefix_lens.tolist() == [0, 0]
    assert split.prompt_metadata.query_start_loc.tolist() == [0, 5, 8]
    assert split.prompt_metadata.mamba_cache_indices.tolist() == [7, 8]
    assert split.prompt_metadata.conv_block_table.squeeze(-1).tolist() == [0, 1, 2]
    assert split.tail_batch.forward_mode == ForwardMode.DECODE
    assert split.tail_batch.batch_size == 3
    assert split.tail_batch.mamba_track_mask is None
    assert split.tail_metadata.query_start_loc.tolist() == [0, 1, 2, 3]
    assert split.tail_metadata.mamba_cache_indices.tolist() == [9, 10, 11]
    # The caller's batch and metadata are untouched.
    assert batch.forward_mode == ForwardMode.MIXED
    assert metadata.mamba_cache_indices.tolist() == [7, 8, 9, 10, 11]


def test_only_a_batch_with_prompts_ahead_of_running_requests_splits() -> None:
    metadata = ForwardMetadata(
        query_start_loc=torch.tensor([0, 1, 2]),
        mamba_cache_indices=torch.tensor([3, 4]),
    )
    prompts_only = mixed_batch([1, 4], [0, 0], ForwardMode.EXTEND)
    tails_only = mixed_batch([1, 1], [10, 20], ForwardMode.EXTEND)
    decode = mixed_batch([1, 1], [10, 20], ForwardMode.DECODE)

    assert decode_tail_request_count(prompts_only) == 0
    assert decode_tail_request_count(decode) == 0
    assert split_decode_tail(prompts_only, metadata) is None
    assert split_decode_tail(tails_only, metadata) is None
    assert split_decode_tail(decode, metadata) is None
