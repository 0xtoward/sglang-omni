# SPDX-License-Identifier: Apache-2.0
"""The batched duplex sampler keeps every session's rules while sampling all rows at once."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.duplex_sampler import (
    build_forbidden_token_index,
    duplex_sample,
    filter_top_k_top_p,
)
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import REQUIRED_SPECIAL_TOKENS
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState

VOCAB = 128


@pytest.fixture
def special():
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[7, 8, 94])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(100, 116))
    ).__getitem__
    return ThinkerAdapter(tokenizer, VOCAB).special


def state(**overrides) -> MiniCPMOThinkerSessionState:
    settings = {"greedy": True, "repetition_penalty": 1.0, **overrides}
    return MiniCPMOThinkerSessionState(sampling=MiniCPMODuplexSampling(**settings))


def sample(logits, units, special):
    return duplex_sample(
        logits,
        units,
        special_tokens=special,
        forbidden_token_index=build_forbidden_token_index(
            special, VOCAB, torch.device("cpu")
        ),
    )


def test_batch_rows_follow_their_own_session_rules(special) -> None:
    logits = torch.full((4, VOCAB), -10.0)
    logits[0, 42] = 5.0
    logits[1, special.chunk_eos] = 5.0
    logits[1, 43] = 4.0
    logits[2, 44] = 5.0
    logits[3, 45] = 5.0
    budget_spent = state(max_new_tokens_per_unit=3)
    units = [
        (state(), 1, False),
        (state(), 1, False),
        (state(), 0, True),
        (budget_spent, 2, False),
    ]
    assert sample(logits, units, special) == [
        42,
        special.chunk_eos,
        special.listen,
        special.chunk_eos,
    ]


def test_forbidden_penalised_and_scaled_tokens_lose_to_competitors(special) -> None:
    logits = torch.full((3, VOCAB), -10.0)
    logits[0, 7] = 5.0
    logits[0, 42] = 4.0
    logits[1, 50] = 5.0
    logits[1, 51] = 4.9
    logits[2, special.listen] = 5.0
    logits[2, 60] = 4.0
    repeated = state(repetition_penalty=1.5)
    repeated.generated_history.extend([50, 50])
    quiet = state(listen_prob_scale=0.5)
    units = [(state(), 1, False), (repeated, 1, False), (quiet, 1, False)]
    assert sample(logits, units, special) == [42, 51, 60]
    assert repeated.generated_history[-1] == 51


def test_listen_mid_turn_becomes_tts_bos_and_turn_eos_ends_the_turn(special) -> None:
    logits = torch.full((2, VOCAB), -10.0)
    logits[0, special.listen] = 5.0
    logits[1, special.turn_eos] = 5.0
    speaking = state()
    speaking.is_turn_ended = False
    ending = state()
    ending.is_turn_ended = False
    assert sample(logits, [(speaking, 1, False), (ending, 1, False)], special) == [
        special.tts_bos,
        special.turn_eos,
    ]
    assert not speaking.is_turn_ended
    assert ending.is_turn_ended


def test_random_rows_sample_inside_their_top_k_and_top_p(special) -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB)
    logits[:, 7] = 50.0
    narrow = state(greedy=False, temperature=0.7, top_k=1, top_p=0.8)
    wide = state(greedy=False, temperature=1.0, top_k=-1, top_p=1.0)
    units = [(narrow, 1, False), (state(), 1, False), (wide, 1, False)]
    picked = sample(logits.clone(), units, special)
    allowed = logits.clone()
    allowed[
        :, build_forbidden_token_index(special, VOCAB, torch.device("cpu"))
    ] = -torch.inf
    assert picked[0] == int(allowed[0].argmax())
    assert picked[1] == int(allowed[1].argmax())
    assert allowed[2, picked[2]] > -torch.inf


def test_filter_top_k_top_p_matches_single_row_filtering() -> None:
    torch.manual_seed(1)
    logits = torch.randn(3, VOCAB)
    filtered = filter_top_k_top_p(logits, top_k=[5, -1, 20], top_p=[1.0, 0.5, 0.9])
    for row, (top_k, top_p) in enumerate([(5, 1.0), (-1, 0.5), (20, 0.9)]):
        expected = filter_top_k_top_p(
            logits[row : row + 1], top_k=[top_k], top_p=[top_p]
        )[0]
        torch.testing.assert_close(filtered[row], expected)
        kept = int((filtered[row] > -torch.inf).sum())
        assert kept <= (top_k if top_k > 0 else VOCAB)
        assert kept >= 1
