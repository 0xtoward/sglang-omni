# SPDX-License-Identifier: Apache-2.0
"""The device duplex sampler keeps every session's rules while sampling all rows at once."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSamplerState,
    DuplexUnitStart,
    build_forbidden_token_index,
    filter_top_k_top_p,
)
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.native_thinker_model_runner import (
    MiniCPMOThinkerModelRunner,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    MiniCPMOSpecialTokenIds,
)
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)

VOCAB = 128


@pytest.fixture
def special() -> MiniCPMOSpecialTokenIds:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[7, 8, 94])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(100, 116))
    ).__getitem__
    return ThinkerAdapter(tokenizer, VOCAB).special


def sampling(**overrides) -> MiniCPMODuplexSampling:
    return MiniCPMODuplexSampling(
        **{"greedy": True, "repetition_penalty": 1.0, **overrides}
    )


class Sessions:
    """Drive the sampler state the way the runner does, one session per slot."""

    def __init__(
        self, special: MiniCPMOSpecialTokenIds, samplings: list[MiniCPMODuplexSampling]
    ) -> None:
        self.special = special
        self.samplings = samplings
        self.state = DuplexSamplerState.allocate(
            len(samplings),
            max(sampling.repetition_window_size for sampling in samplings),
            VOCAB,
            torch.device("cpu"),
        )
        self.forbidden_token_index = build_forbidden_token_index(
            special, VOCAB, torch.device("cpu")
        )
        self.is_new_session = [True] * len(samplings)

    def start(
        self,
        slots: list[int],
        generation_steps: list[int] | None = None,
        listen_forced: list[bool] | None = None,
    ) -> None:
        self.state.start_units(
            [
                DuplexUnitStart(
                    sampling_slot=slot,
                    sampling=self.samplings[slot],
                    is_new_session=self.is_new_session[slot],
                    generation_steps=(generation_steps or [1] * len(slots))[row],
                    is_listen_forced=(listen_forced or [False] * len(slots))[row],
                )
                for row, slot in enumerate(slots)
            ]
        )
        for slot in slots:
            self.is_new_session[slot] = False

    def reopen(self, slot: int) -> None:
        self.is_new_session[slot] = True

    def sample(self, logits: torch.Tensor, slots: list[int]) -> list[int]:
        return self.state.sample(
            logits,
            torch.tensor(slots),
            [self.samplings[slot] for slot in slots],
            special_tokens=self.special,
            forbidden_token_index=self.forbidden_token_index,
        ).tolist()


def peaked(*peaks: dict[int, float]) -> torch.Tensor:
    logits = torch.full((len(peaks), VOCAB), -10.0)
    for row, values in enumerate(peaks):
        for token_id, value in values.items():
            logits[row, token_id] = value
    return logits


def test_batch_rows_follow_their_own_session_rules(special) -> None:
    sessions = Sessions(
        special,
        [sampling(), sampling(), sampling(), sampling(max_new_tokens_per_unit=3)],
    )
    sessions.start(
        [0, 1, 2, 3],
        generation_steps=[1, 1, 0, 2],
        listen_forced=[False, False, True, False],
    )
    logits = peaked({42: 5.0}, {special.chunk_eos: 5.0, 43: 4.0}, {44: 5.0}, {45: 5.0})
    assert sessions.sample(logits, [0, 1, 2, 3]) == [
        42,
        special.chunk_eos,
        special.listen,
        special.chunk_eos,
    ]


def test_forbidden_penalised_and_scaled_tokens_lose_to_competitors(special) -> None:
    sessions = Sessions(
        special,
        [sampling(), sampling(repetition_penalty=1.5), sampling(listen_prob_scale=0.5)],
    )
    sessions.start([1])
    assert sessions.sample(peaked({50: 5.0}), [1]) == [50]
    sessions.start([0, 1, 2])
    logits = peaked(
        {7: 5.0, 42: 4.0}, {50: 5.0, 51: 4.9}, {special.listen: 5.0, 60: 4.0}
    )
    assert sessions.sample(logits, [0, 1, 2]) == [42, 51, 60]
    assert sessions.sample(peaked({51: 5.0, 52: 4.9}), [1]) == [52]


def test_listen_mid_turn_becomes_tts_bos_and_turn_eos_ends_the_turn(special) -> None:
    sessions = Sessions(special, [sampling(), sampling()])
    sessions.start([0, 1])
    assert sessions.sample(peaked({42: 5.0}, {43: 5.0}), [0, 1]) == [42, 43]
    logits = peaked({special.listen: 5.0}, {special.turn_eos: 5.0})
    assert sessions.sample(logits, [0, 1]) == [special.tts_bos, special.turn_eos]
    logits = peaked({special.listen: 5.0}, {special.listen: 5.0})
    assert sessions.sample(logits, [0, 1]) == [special.tts_bos, special.listen]


def test_random_rows_sample_inside_their_top_k_and_top_p(special) -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB)
    logits[:, 7] = 50.0
    sessions = Sessions(
        special,
        [
            sampling(greedy=False, temperature=0.7, top_k=1, top_p=0.8),
            sampling(),
            sampling(greedy=False, temperature=1.0, top_k=-1, top_p=1.0),
        ],
    )
    sessions.start([0, 1, 2])
    picked = sessions.sample(logits.clone(), [0, 1, 2])
    allowed = logits.clone()
    allowed[:, build_forbidden_token_index(special, VOCAB, torch.device("cpu"))] = (
        -torch.inf
    )
    assert picked[0] == int(allowed[0].argmax())
    assert picked[1] == int(allowed[1].argmax())
    assert allowed[2, picked[2]] > -torch.inf


def filter_rows(logits, settings):
    top_k = [k for k, _ in settings]
    top_p = [p for _, p in settings]
    return filter_top_k_top_p(
        logits,
        top_k=torch.tensor(top_k),
        top_p=torch.tensor(top_p),
        max_top_k=max((k for k in top_k if 0 < k < VOCAB), default=0),
        has_top_p=any(0.0 < p < 1.0 for p in top_p),
    )


def test_filter_top_k_top_p_matches_single_row_filtering() -> None:
    torch.manual_seed(1)
    logits = torch.randn(3, VOCAB)
    settings = [(5, 1.0), (-1, 0.5), (20, 0.9)]
    filtered = filter_rows(logits, settings)
    for row, (top_k, top_p) in enumerate(settings):
        expected = filter_rows(logits[row : row + 1], [(top_k, top_p)])[0]
        torch.testing.assert_close(filtered[row], expected)
        kept = int((filtered[row] > -torch.inf).sum())
        assert kept <= (top_k if top_k > 0 else VOCAB)
        assert kept >= 1


def test_rows_ending_the_chunk_keep_their_history(special) -> None:
    sessions = Sessions(special, [sampling(repetition_penalty=1.5)])
    sessions.start([0])
    logits = peaked({special.chunk_eos: 5.0, 30: 4.0})
    assert sessions.sample(logits, [0]) == [special.chunk_eos]
    sessions.start([0])
    assert sessions.sample(peaked({30: 5.0, 31: 4.9}), [0]) == [30]


def test_rows_with_top_k_sample_among_their_top_candidates(special) -> None:
    torch.manual_seed(2)
    logits = torch.randn(4, VOCAB)
    # Control tokens never win, so every pick comes from the second stage unchanged.
    logits[:, 100:116] = -50.0
    samplings = [
        sampling(greedy=False, temperature=0.9, top_k=k, top_p=p)
        for k, p in ((3, 1.0), (5, 0.6), (1, 0.8), (20, 0.95))
    ]
    allowed = logits.clone()
    allowed[:, build_forbidden_token_index(special, VOCAB, torch.device("cpu"))] = (
        -torch.inf
    )
    for trial in range(20):
        # A fresh session per trial keeps earlier picks out of the penalty window.
        sessions = Sessions(special, samplings)
        sessions.start([0, 1, 2, 3])
        picked = sessions.sample(logits.clone(), [0, 1, 2, 3])
        for row, (token_id, settings) in enumerate(zip(picked, samplings)):
            top = torch.topk(allowed[row], settings.top_k).indices.tolist()
            assert token_id in top


def test_history_spans_units_up_to_the_window(special) -> None:
    sessions = Sessions(
        special, [sampling(repetition_penalty=2.0, repetition_window_size=2)]
    )
    for token_id in (10, 11, 12):
        sessions.start([0])
        assert sessions.sample(peaked({token_id: 5.0}), [0]) == [token_id]
    sessions.start([0])
    assert sessions.sample(peaked({10: 5.0, 9: 4.0}), [0]) == [10]
    assert sessions.sample(peaked({12: 5.0, 9: 4.0}), [0]) == [9]


def test_a_reused_slot_starts_a_fresh_session(special) -> None:
    sessions = Sessions(special, [sampling(repetition_penalty=2.0)])
    sessions.start([0])
    assert sessions.sample(peaked({42: 5.0}), [0]) == [42]
    sessions.reopen(0)
    sessions.start([0])
    logits = peaked({special.listen: 5.0})
    assert sessions.sample(logits, [0]) == [special.listen]
    sessions.start([0])
    assert sessions.sample(peaked({42: 5.0, 41: 4.0}), [0]) == [42]


def test_a_step_drawn_past_the_unit_end_changes_nothing(special) -> None:
    sessions = Sessions(special, [sampling(repetition_penalty=2.0)])
    sessions.start([0])
    assert sessions.sample(peaked({special.chunk_eos: 5.0}), [0]) == [special.chunk_eos]
    sessions.sample(peaked({42: 5.0}), [0])
    sessions.start([0])
    assert sessions.sample(peaked({special.listen: 5.0}), [0]) == [special.listen]
    sessions.start([0])
    assert sessions.sample(peaked({42: 5.0, 41: 4.0}), [0]) == [42]


def test_talker_conditions_keep_each_step_hidden_state_until_the_unit_ends(
    special,
) -> None:
    runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
    runner.special_tokens = special
    runner.pending_hidden = {}
    data = SimpleNamespace(
        generation_steps=0,
        pending_unit_token=None,
        generated_unit_ids=[],
        talker_conditions=[],
    )
    # One buffer for every step, as a replayed decode graph returns.
    hidden_buffer = torch.zeros(1, 4)
    for step, token_id in enumerate((42, 43, 44, special.chunk_eos)):
        hidden_buffer.fill_(float(step))
        data.generation_steps = step
        runner.post_process_outputs(
            None,
            SchedulerOutput(
                requests=[SchedulerRequest(request_id="unit", data=data)],
                batch_data=None,
            ),
            {
                "unit": RequestOutput(
                    request_id="unit",
                    data=token_id,
                    extra={"hidden_states": hidden_buffer},
                )
            },
        )
    runner.on_request_finished("unit", data)
    assert data.generated_unit_ids == [43, 44]
    assert [
        (token_id, ends_turn) for token_id, _, ends_turn in data.talker_conditions
    ] == [
        (43, False),
        (44, False),
    ]
    assert [float(hidden[0]) for _, hidden, _ in data.talker_conditions] == [2.0, 3.0]
