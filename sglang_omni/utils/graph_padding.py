# SPDX-License-Identifier: Apache-2.0
"""Pad bucket-miss batches onto captured CUDA graphs of aux modules.

The sglang decode runner never demands an exact batch match: it bisects to
the smallest captured bucket >= the live batch and points the filler rows at
reserved storage (``memory_pool.py``: "The padded slot 0 is used for writing
dummy outputs from padded tokens"). Aux-module graphs in sglang-omni (acoustic
tails, vocoders, encoders) each hand-roll their own selection; modules whose
graphs gather *and scatter per-row persistent state* (slot pools) cannot pad
naively — a filler row would overwrite a live request's state — so they tend
to ship exact-match selection with an eager fallback instead, and every
off-bucket step silently pays the eager launch tax.

This module holds the shared mechanics of the safe alternative:

* :func:`select_padded_graph` — bisect-style bucket selection over a captured
  graph dict keyed by ``(batch_size, capacity)``, with an optional exclusion
  for positional (non-gather) captures and an optional overlay dict of
  gather-mode twins.
* :func:`pad_rows` — extend per-row input tensors with filler rows.

The caller stays responsible for the policy half: choosing the sacrificial
storage the filler rows point at (a currently-free slot, or — the sglang
idiom, preferable for new adopters — a dedicated reserved row allocated
beyond the live slot range), and slicing the graph output back to the live
row count. See ``models/dots_tts/tail.py`` for the first adopter.
"""

from __future__ import annotations

from typing import TypeVar

import torch

GraphT = TypeVar("GraphT")


def select_padded_graph(
    graphs: dict[tuple[int, int], GraphT],
    rows: int,
    capacity: int,
    *,
    skip_batch: int | None = None,
    extra: dict[tuple[int, int], GraphT] | None = None,
) -> tuple[GraphT | None, int]:
    """Pick the smallest captured graph a ``rows``-row batch can pad up to.

    Candidates need ``batch_size > rows`` and ``bucket_capacity >= capacity``;
    ties resolve to the smallest batch then the smallest capacity. Entries in
    ``graphs`` whose batch equals ``skip_batch`` are ignored (captures that
    read state positionally instead of via the slot-index input buffer);
    ``extra`` supplies gather-mode replacements for such batches. Returns
    ``(graph, filler_row_count)`` or ``(None, 0)``.
    """
    pool = [
        (batch_size, bucket_capacity, graphs)
        for batch_size, bucket_capacity in graphs
        if batch_size > rows
        and bucket_capacity >= capacity
        and batch_size != skip_batch
    ]
    if extra:
        pool += [
            (batch_size, bucket_capacity, extra)
            for batch_size, bucket_capacity in extra
            if batch_size > rows and bucket_capacity >= capacity
        ]
    if not pool:
        return None, 0
    batch_size, bucket_capacity, source = min(
        pool, key=lambda item: (item[0], item[1])
    )
    return source[(batch_size, bucket_capacity)], batch_size - rows


def pad_rows(
    tensor: torch.Tensor,
    pad: int,
    *,
    fill_value: int | float | None = None,
) -> torch.Tensor:
    """Append ``pad`` filler rows to a per-row input tensor.

    Filler rows are zeros unless ``fill_value`` is given (e.g. the sacrificial
    slot index for the slot-id tensor). The result matches the captured input
    buffer's batch dimension, so a plain ``copy_`` stages it for replay.
    """
    if pad <= 0:
        return tensor
    shape = (pad, *tensor.shape[1:])
    filler = (
        tensor.new_zeros(shape)
        if fill_value is None
        else tensor.new_full(shape, fill_value)
    )
    return torch.cat([tensor, filler])
