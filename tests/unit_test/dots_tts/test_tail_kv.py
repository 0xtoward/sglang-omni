# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

tail_kv = pytest.importorskip("sglang_omni.models.dots_tts.tail_kv")


@pytest.mark.accelerator
@pytest.mark.parametrize(("num_slots", "tokens"), [(8, 0), (12, 5), (24, 13)])
def test_kv_copies_preserve_live_rows_and_skip_dummy_storage(
    num_slots: int, tokens: int
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton KV copies")
    else:
        pass

    torch.manual_seed(0)
    pool_shape = (2, num_slots, 3, 23, 11)
    pool_backing = [
        torch.randn(pool_shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    pools = [backing[..., 1:20, 1:8] for backing in pool_backing]
    out_shape = (2, 6, 3, tokens + 2, 9)
    out_backing = [
        torch.full(out_shape, value, device="cuda", dtype=torch.bfloat16)
        for value in (123, -123)
    ]
    outputs = [backing[..., 1 : 1 + tokens, 1:8] for backing in out_backing]
    expected_outputs = [backing.clone() for backing in out_backing]
    expected_pools = [backing.clone() for backing in pool_backing]
    slot_ids = [num_slots - 1, num_slots, 1, num_slots, 0, num_slots // 2]
    start_ids = [0, 10_000, 3, -10_000, 4, 5]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.long)
    starts = torch.tensor(start_ids, device="cuda", dtype=torch.long)
    valid_lengths = torch.full_like(slots, tokens)

    # note (0xtoward): Strided views and repeated dummy IDs exercise masked
    # accesses; checking the backing tensors also catches writes outside views.
    tail_kv.gather_kv(*pools, slots, valid_lengths, *outputs)
    for pool, expected in zip(pools, expected_outputs):
        view = expected[..., 1 : 1 + tokens, 1:8]
        for row, slot in enumerate(slot_ids):
            if slot < num_slots:
                view[:, row].copy_(pool[:, slot, :, :tokens])
            else:
                view[:, row].zero_()
    for actual, expected in zip(out_backing, expected_outputs):
        assert torch.equal(actual, expected)

    tail_kv.scatter_kv(*outputs, slots, starts, *pools)
    for output, expected in zip(outputs, expected_pools):
        view = expected[..., 1:20, 1:8]
        for row, (slot, start) in enumerate(zip(slot_ids, start_ids)):
            if slot < num_slots:
                view[:, slot, :, start : start + tokens].copy_(output[:, row])
            else:
                pass
    for actual, expected in zip(pool_backing, expected_pools):
        assert torch.equal(actual, expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("tokens", [0, 13])
def test_gather_valid_prefixes_clear_poisoned_suffixes_on_graph_replay(
    tokens: int,
) -> None:
    """Replay with reordered rows and changing lengths must clear stale scratch KV."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton KV copies")
    else:
        pass

    torch.manual_seed(0)
    pool_shape = (2, 8, 3, tokens + 3, 11)
    pool_backing = [
        torch.randn(pool_shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    pools = [backing[..., 1 : 1 + tokens, 1:8] for backing in pool_backing]
    output_shape = (2, 6, 3, tokens + 2, 9)
    output_backing = [
        torch.full(output_shape, 123, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    outputs = [backing[..., 1 : 1 + tokens, 1:8] for backing in output_backing]
    expected_outputs = [backing.clone() for backing in output_backing]
    slots = torch.tensor([7, 8, 1, 8, 0, 4], device="cuda", dtype=torch.long)
    valid_lengths = torch.full_like(slots, tokens)
    tail_kv.gather_kv(*pools, slots, valid_lengths, *outputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tail_kv.gather_kv(*pools, slots, valid_lengths, *outputs)

    replay_rows = [
        ([7, 8, 1, 8, 0, 4], [tokens, 0, tokens // 2, 0, 0, tokens // 3]),
        ([4, 8, 0, 8, 7, 1], [0, 0, tokens, 0, tokens // 3, tokens]),
        ([7, 8, 1, 8, 0, 4], [tokens, 0, tokens // 2, 0, 0, tokens // 3]),
    ]
    for slot_ids, length_ids in replay_rows:
        slots.copy_(torch.tensor(slot_ids, device="cuda", dtype=torch.long))
        valid_lengths.copy_(torch.tensor(length_ids, device="cuda", dtype=torch.long))
        for pool, backing in zip(pools, pool_backing):
            backing.normal_()
            for slot, valid_length in zip(slot_ids, length_ids):
                if slot < pool.size(1):
                    pool[:, slot, :, valid_length:].fill_(float("nan"))
                else:
                    pass
        for output in outputs:
            output.fill_(float("nan"))
        graph.replay()
        for pool, expected in zip(pools, expected_outputs):
            view = expected[..., 1 : 1 + tokens, 1:8]
            view.zero_()
            for row, (slot, valid_length) in enumerate(zip(slot_ids, length_ids)):
                if slot < pool.size(1):
                    view[:, row, :, :valid_length].copy_(
                        pool[:, slot, :, :valid_length]
                    )
                else:
                    pass
        for actual, expected in zip(output_backing, expected_outputs):
            assert torch.equal(actual, expected)


@pytest.mark.accelerator
@pytest.mark.parametrize("backend", [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
def test_gather_valid_prefix_preserves_same_shape_attention_output(
    backend: SDPBackend,
) -> None:
    """Masking finite trailing KV preserves attention with unchanged key dimensions."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Triton KV copies")
    else:
        pass
    torch.manual_seed(0)
    pool_shape = (2, 8, 3, 13, 16)
    pools = [
        torch.randn(pool_shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    slot_ids = [7, 8, 1, 8, 0, 4]
    length_ids = [13, 0, 7, 0, 0, 3]
    slots = torch.tensor(slot_ids, device="cuda", dtype=torch.long)
    valid_lengths = torch.tensor(length_ids, device="cuda", dtype=torch.long)
    outputs = [
        torch.empty((2, 6, 3, 13, 16), device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    reference = [torch.zeros_like(output) for output in outputs]
    for pool, prefix in zip(pools, reference):
        for row, slot in enumerate(slot_ids):
            if slot < pool.size(1):
                prefix[:, row].copy_(pool[:, slot])
            else:
                pass
    tail_kv.gather_kv(*pools, slots, valid_lengths, *outputs)
    query = torch.randn((2, 6, 3, 2, 16), device="cuda", dtype=torch.bfloat16)
    current = [torch.randn_like(query) for _ in range(2)]
    mask = torch.arange(13, device="cuda").reshape(1, 1, 1, 13) < valid_lengths.reshape(
        6, 1, 1, 1
    )
    mask = torch.cat(
        [mask, torch.ones((6, 1, 1, 2), device="cuda", dtype=torch.bool)], dim=-1
    )
    with sdpa_kernel(backend):
        for layer in range(2):
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                query[layer],
                torch.cat([reference[0][layer], current[0][layer]], dim=-2),
                torch.cat([reference[1][layer], current[1][layer]], dim=-2),
                attn_mask=mask,
            )
            actual_output = torch.nn.functional.scaled_dot_product_attention(
                query[layer],
                torch.cat([outputs[0][layer], current[0][layer]], dim=-2),
                torch.cat([outputs[1][layer], current[1][layer]], dim=-2),
                attn_mask=mask,
            )
            torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)
