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
    # test only (xp): per retained finished unit, whether the user was quiet and the model listened
    unit_quiet: list[bool] = field(default_factory=list)
    # test only (xp): 0 user quiet and model listening, 1 model speaking while the user is quiet, 2 user speaking
    unit_kinds: list[int] = field(default_factory=list)

    def drop_units(
        self,
        next_unit_start: int,
        target_start: int,
        *,
        policy: str,
        protect_tokens: int,
        unit_scores: list[float] | None = None,
    ) -> list[tuple[int, int]] | None:
        """test only (xp): choose finished units to drop until the next unit starts at or below target_start.

        basic drops the oldest; quiet drops quiet units first; knorm / keydiff drop the highest-scored units
        first. Units starting within protect_tokens of the next unit go only in the oldest-first fallback.
        """
        count = len(self.unit_starts)
        need = next_unit_start - target_start
        if count < 2 or need <= 0:
            return None
        else:
            pass
        lengths = [
            self.unit_starts[index + 1] - self.unit_starts[index]
            for index in range(count - 1)
        ]
        eligible = [
            index
            for index in range(count - 1)
            if self.unit_starts[index] < next_unit_start - protect_tokens
        ]
        if policy == "quiet":
            order = [index for index in eligible if self.unit_quiet[index]]
        elif policy == "user":
            order = [index for index in eligible if self.unit_kinds[index] == 0] + [
                index for index in eligible if self.unit_kinds[index] == 1
            ]
        elif policy in ("knorm", "keydiff") and unit_scores is not None:
            order = sorted(eligible, key=lambda index: -unit_scores[index])
        elif policy in ("quietkd", "quietkn") and unit_scores is not None:
            order = [index for index in eligible if self.unit_quiet[index]] + sorted(
                (index for index in eligible if not self.unit_quiet[index]),
                key=lambda index: -unit_scores[index],
            )
        else:
            order = []
        dropped: set[int] = set()
        total = 0
        for index in [*order, *range(count - 1)]:
            if total >= need:
                break
            elif index not in dropped:
                dropped.add(index)
                total += lengths[index]
            else:
                pass
        if not dropped:
            return None
        else:
            pass
        ranges: list[tuple[int, int]] = []
        for index in sorted(dropped):
            start, end = self.unit_starts[index], self.unit_starts[index + 1]
            if ranges and ranges[-1][1] == start:
                ranges[-1] = (ranges[-1][0], end)
            else:
                ranges.append((start, end))
        starts: list[int] = []
        quiet: list[bool] = []
        kinds: list[int] = []
        shift = 0
        for index in range(count):
            if index in dropped:
                shift += lengths[index]
            else:
                starts.append(self.unit_starts[index] - shift)
                quiet.append(self.unit_quiet[index])
                kinds.append(self.unit_kinds[index])
        self.unit_starts = starts
        self.unit_quiet = quiet
        self.unit_kinds = kinds
        return ranges

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
