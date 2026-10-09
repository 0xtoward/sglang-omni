# SPDX-License-Identifier: Apache-2.0
"""The batched duplex sampler keeps every session's rules while sampling all rows at once."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSampleRow,
    build_forbidden_token_index,
    duplex_sample,
    filter_top_k_top_p,
)
from sglang_omni.models.minicpm_o.native_config import MiniCPMODuplexSampling
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    MiniCPMOSpecialTokenIds,
)
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState

VOCAB_SIZE = 128


@pytest.fixture
def special_tokens() -> MiniCPMOSpecialTokenIds:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=[7, 8, 94])
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(100, 116))
    ).__getitem__
    return ThinkerAdapter(tokenizer, VOCAB_SIZE).special


def session_state(
    **sampling_overrides: float | int | bool,
) -> MiniCPMOThinkerSessionState:
    settings = {"greedy": True, "repetition_penalty": 1.0, **sampling_overrides}
    return MiniCPMOThinkerSessionState(sampling=MiniCPMODuplexSampling(**settings))


def unit_row(
    state: MiniCPMOThinkerSessionState,
    generation_step: int = 1,
    is_listen_forced: bool = False,
) -> DuplexSampleRow:
    return DuplexSampleRow(
        thinker_state=state,
        generation_step=generation_step,
        is_listen_forced=is_listen_forced,
    )


def forbidden_token_index(special_tokens: MiniCPMOSpecialTokenIds) -> torch.Tensor:
    return build_forbidden_token_index(special_tokens, VOCAB_SIZE, torch.device("cpu"))


def sample(
    logits: torch.Tensor,
    rows: list[DuplexSampleRow],
    special_tokens: MiniCPMOSpecialTokenIds,
) -> list[int]:
    return duplex_sample(
        logits,
        rows,
        special_tokens=special_tokens,
        forbidden_token_index=forbidden_token_index(special_tokens),
    )


def test_batch_rows_follow_their_own_session_rules(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((4, VOCAB_SIZE), -10.0)
    logits[0, 42] = 5.0
    logits[1, special_tokens.chunk_eos] = 5.0
    logits[1, 43] = 4.0
    logits[2, 44] = 5.0
    logits[3, 45] = 5.0
    rows = [
        unit_row(session_state()),
        unit_row(session_state()),
        unit_row(session_state(), generation_step=0, is_listen_forced=True),
        unit_row(session_state(max_new_tokens_per_unit=3), generation_step=2),
    ]
    assert sample(logits, rows, special_tokens) == [
        42,
        special_tokens.chunk_eos,
        special_tokens.listen,
        special_tokens.chunk_eos,
    ]


def test_forbidden_penalized_and_scaled_tokens_lose_to_competitors(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((3, VOCAB_SIZE), -10.0)
    logits[0, 7] = 5.0
    logits[0, 42] = 4.0
    logits[1, 50] = 5.0
    logits[1, 51] = 4.9
    logits[2, special_tokens.listen] = 5.0
    logits[2, 60] = 4.0
    repeated = session_state(repetition_penalty=1.5)
    repeated.generated_history.extend([50, 50])
    quiet = session_state(listen_prob_scale=0.5)
    rows = [unit_row(session_state()), unit_row(repeated), unit_row(quiet)]
    assert sample(logits, rows, special_tokens) == [42, 51, 60]
    assert repeated.generated_history[-1] == 51


def test_listen_mid_turn_becomes_tts_bos_and_turn_eos_ends_the_turn(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((2, VOCAB_SIZE), -10.0)
    logits[0, special_tokens.listen] = 5.0
    logits[1, special_tokens.turn_eos] = 5.0
    speaking = session_state()
    speaking.is_turn_ended = False
    ending = session_state()
    ending.is_turn_ended = False
    rows = [unit_row(speaking), unit_row(ending)]
    assert sample(logits, rows, special_tokens) == [
        special_tokens.tts_bos,
        special_tokens.turn_eos,
    ]
    assert not speaking.is_turn_ended
    assert ending.is_turn_ended


def test_random_rows_sample_inside_their_top_k_and_top_p(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB_SIZE)
    logits[:, 7] = 50.0
    narrow = session_state(greedy=False, temperature=0.7, top_k=1, top_p=0.8)
    wide = session_state(greedy=False, temperature=1.0, top_k=-1, top_p=1.0)
    rows = [unit_row(narrow), unit_row(session_state()), unit_row(wide)]
    token_ids = sample(logits.clone(), rows, special_tokens)
    allowed_logits = logits.clone()
    allowed_logits[:, forbidden_token_index(special_tokens)] = -torch.inf
    assert token_ids[0] == int(allowed_logits[0].argmax())
    assert token_ids[1] == int(allowed_logits[1].argmax())
    assert allowed_logits[2, token_ids[2]] > -torch.inf


def filter_rows(
    logits: torch.Tensor, settings: list[tuple[int, float]]
) -> torch.Tensor:
    top_k = [row_top_k for row_top_k, _ in settings]
    top_p = [row_top_p for _, row_top_p in settings]
    return filter_top_k_top_p(
        logits,
        top_k=torch.tensor(top_k),
        top_p=torch.tensor(top_p),
        max_top_k=max(
            (row_top_k for row_top_k in top_k if 0 < row_top_k < VOCAB_SIZE),
            default=0,
        ),
        has_top_p=any(0.0 < row_top_p < 1.0 for row_top_p in top_p),
    )


def test_filter_top_k_top_p_matches_single_row_filtering() -> None:
    torch.manual_seed(1)
    logits = torch.randn(3, VOCAB_SIZE)
    settings = [(5, 1.0), (-1, 0.5), (20, 0.9)]
    filtered_logits = filter_rows(logits, settings)
    for row, (top_k, top_p) in enumerate(settings):
        expected = filter_rows(logits[row : row + 1], [(top_k, top_p)])[0]
        torch.testing.assert_close(filtered_logits[row], expected)
        kept_count = int((filtered_logits[row] > -torch.inf).sum())
        assert kept_count <= (top_k if top_k > 0 else VOCAB_SIZE)
        assert kept_count >= 1


def test_rows_ending_the_chunk_keep_their_history(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((2, VOCAB_SIZE), -10.0)
    logits[0, special_tokens.chunk_eos] = 5.0
    logits[1, 42] = 5.0
    ended = session_state(repetition_penalty=1.5)
    ended.generated_history.extend([30, 31])
    speaking = session_state(repetition_penalty=1.5)
    rows = [unit_row(ended), unit_row(speaking)]
    assert sample(logits, rows, special_tokens) == [special_tokens.chunk_eos, 42]
    assert ended.generated_history == [30, 31]
    assert speaking.generated_history == [42]


def test_rows_with_top_k_sample_among_their_top_candidates(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    torch.manual_seed(2)
    logits = torch.randn(4, VOCAB_SIZE)
    # note (0xtoward): control tokens never win, so every pick comes from the second stage unchanged.
    logits[:, 100:116] = -50.0
    states = [
        session_state(greedy=False, temperature=0.9, top_k=top_k, top_p=top_p)
        for top_k, top_p in ((3, 1.0), (5, 0.6), (1, 0.8), (20, 0.95))
    ]
    allowed_logits = logits.clone()
    allowed_logits[:, forbidden_token_index(special_tokens)] = -torch.inf
    for _ in range(20):
        token_ids = sample(
            logits.clone(), [unit_row(state) for state in states], special_tokens
        )
        for row, (token_id, state) in enumerate(zip(token_ids, states, strict=True)):
            top_token_ids = torch.topk(
                allowed_logits[row], state.sampling.top_k
            ).indices
            assert token_id in top_token_ids.tolist()
