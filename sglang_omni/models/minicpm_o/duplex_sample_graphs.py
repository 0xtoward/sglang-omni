# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs of the MiniCPM-o duplex sampling step."""

from __future__ import annotations

import torch

from sglang_omni.models.minicpm_o.duplex_sampler import DuplexSamplerState
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.special_tokens import MiniCPMOSpecialTokenIds
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph


class DuplexSampleGraphs:
    """One graph of the duplex sampling step per decode graph batch size.

    Every size is captured before serving for rows whose top-k is at most top_k; the padded
    rows of a replay sample on the state's spare row.
    """

    def __init__(
        self,
        state: DuplexSamplerState,
        batch_sizes: list[int],
        top_k: int,
        *,
        special_tokens: MiniCPMOSpecialTokenIds,
        forbidden_token_index: torch.Tensor,
        backend: DeviceGraphBackend,
    ) -> None:
        self.state = state
        self.top_k = top_k
        self.special_tokens = special_tokens
        self.forbidden_token_index = forbidden_token_index
        self.batch_sizes = sorted(set(batch_sizes))
        largest_batch_size = self.batch_sizes[-1]
        device = state.history.device
        self.logits = torch.zeros(
            largest_batch_size, state.vocab, dtype=torch.float32, device=device
        )
        self.slots = torch.full(
            (largest_batch_size,), state.spare, dtype=torch.long, device=device
        )
        self.next_token_ids = torch.zeros(
            largest_batch_size, dtype=torch.long, device=device
        )
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        pool = backend.graph_pool_handle()
        stream = device_module.Stream(device=device)
        # note (0xtoward): largest first on one stream, so smaller graphs reuse its pool.
        for batch_size in reversed(self.batch_sizes):
            logits = self.logits[:batch_size]
            slots = self.slots[:batch_size]
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                # note (0xtoward): warm-ups on the spare row settle lazy kernel state.
                for _ in range(2):
                    self.draw(logits, slots)
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.next_token_ids[:batch_size].copy_(self.draw(logits, slots))
            device_module.current_stream(device).wait_stream(stream)
            self.graphs[batch_size] = graph

    def draw(self, logits: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        return self.state.draw(
            logits,
            slots,
            max_top_k=self.top_k,
            has_top_p=True,
            is_top_k_everywhere=True,
            special_tokens=self.special_tokens,
            forbidden_token_index=self.forbidden_token_index,
        )

    def fits(self, samplings: list[MiniCPMODuplexSampling]) -> bool:
        """Whether a batch of these sessions can replay a graph."""
        return len(samplings) <= self.batch_sizes[-1] and all(
            0 < sampling.top_k <= self.top_k for sampling in samplings
        )

    def sample(self, logits: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        count = len(slots)
        batch_size = next(size for size in self.batch_sizes if size >= count)
        self.logits[:count].copy_(logits)
        self.slots[:count].copy_(slots)
        self.slots[count:batch_size].fill_(self.state.spare)
        self.graphs[batch_size].replay()
        return self.next_token_ids[:count].clone()
