# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o thinker session history and bounded unit request state."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.scheduling.sglang_backend.ar_session import HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


@dataclass(kw_only=True)
class MiniCPMOThinkerSessionState:
    sampling: MiniCPMODuplexSampling
    is_turn_ended: bool = True
    is_prefix_pending: bool = True
    force_listen_counter: int = 0
    generated_history: list[int] = field(default_factory=list)
    # note (0xtoward): Session position of each retained finished unit's <unit>; the first one ends the system prompt and reference audio.
    unit_starts: list[int] = field(default_factory=list)

    def drop_oldest_units(
        self, next_unit_start: int, target_start: int
    ) -> HistoryCut | None:
        """Drop the oldest finished units until the next unit starts at or below target_start.

        The newest finished unit stays: its generated tokens are not retained ids yet.
        """
        dropped_unit_count = 0
        while dropped_unit_count + 1 < len(self.unit_starts) and (
            next_unit_start
            - (self.unit_starts[dropped_unit_count] - self.unit_starts[0])
            > target_start
        ):
            dropped_unit_count += 1
        if dropped_unit_count == 0:
            return None
        else:
            cut = HistoryCut(
                start=self.unit_starts[0], end=self.unit_starts[dropped_unit_count]
            )
            self.unit_starts = [
                unit_start - (cut.end - cut.start)
                for unit_start in self.unit_starts[dropped_unit_count:]
            ]
            return cut


@dataclass(kw_only=True)
class DuplexUnitRequestData(SGLangARRequestData):
    """One bounded generated unit appended to an SGLang streaming session."""

    thinker_state: MiniCPMOThinkerSessionState
    # note (0xtoward): Index of the unit's <unit> among its own token ids.
    unit_token_index: int
    # note (Junnan Li): Each entry is (token id, thinker hidden state, ends the turn).
    talker_conditions: list[tuple[int, torch.Tensor, bool]] = field(
        default_factory=list
    )
    generated_unit_ids: list[int] = field(default_factory=list)
    pending_unit_token: int | None = None
    is_listen_forced: bool = False
    enforce_request_limits: bool = True
