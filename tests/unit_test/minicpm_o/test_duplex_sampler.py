# SPDX-License-Identifier: Apache-2.0
"""The batched duplex sampler keeps every session's rules while sampling all rows at once."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.duplex_sampler import (
    DuplexSampleRow,
    build_forbidden_token_index,
    duplex_sample,
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
from sglang_omni.models.minicpm_o.thinker_state import MiniCPMOThinkerSessionState
from tests.unit_test.fixtures.accelerator import require_cuda

VOCAB_SIZE = 128
BAD_TOKEN_IDS = [7, 8, 94]
FIRST_SPECIAL_TOKEN_ID = 100


def make_tokenizer() -> Mock:
    tokenizer = Mock(unk_token_id=0, bad_token_ids=BAD_TOKEN_IDS)
    tokenizer.convert_tokens_to_ids.side_effect = dict(
        zip(REQUIRED_SPECIAL_TOKENS, range(FIRST_SPECIAL_TOKEN_ID, VOCAB_SIZE))
    ).__getitem__
    return tokenizer


@pytest.fixture
def special_tokens() -> MiniCPMOSpecialTokenIds:
    return ThinkerAdapter(make_tokenizer(), VOCAB_SIZE).special


def make_session_state(
    **sampling_overrides: float | int | bool,
) -> MiniCPMOThinkerSessionState:
    settings = {"greedy": True, "repetition_penalty": 1.0, **sampling_overrides}
    return MiniCPMOThinkerSessionState(sampling=MiniCPMODuplexSampling(**settings))


def make_unit_row(
    state: MiniCPMOThinkerSessionState,
    generation_step: int = 1,
    is_listen_forced: bool = False,
) -> DuplexSampleRow:
    return DuplexSampleRow(
        thinker_state=state,
        generation_step=generation_step,
        is_listen_forced=is_listen_forced,
    )


def sample(
    logits: torch.Tensor,
    rows: list[DuplexSampleRow],
    special_tokens: MiniCPMOSpecialTokenIds,
) -> list[int]:
    return duplex_sample(
        logits,
        rows,
        special_tokens=special_tokens,
        forbidden_token_index=build_forbidden_token_index(
            special_tokens, VOCAB_SIZE, logits.device
        ),
    )


def filter_single_row(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """The per-row top-k and top-p filter the duplex sampler applied before batching."""
    filtered_logits = logits.clone()
    if 0 < top_k < filtered_logits.numel():
        threshold = torch.topk(filtered_logits, top_k).values[-1]
        filtered_logits[filtered_logits < threshold] = -torch.inf
    else:
        pass
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            filtered_logits, descending=True, stable=True
        )
        cumulative_probabilities = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )
        should_remove = cumulative_probabilities > top_p
        should_remove[1:] = should_remove[:-1].clone()
        should_remove[0] = False
        filtered_logits[sorted_indices[should_remove]] = -torch.inf
    else:
        pass
    return filtered_logits


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
        make_unit_row(make_session_state()),
        make_unit_row(make_session_state()),
        make_unit_row(make_session_state(), generation_step=0, is_listen_forced=True),
        make_unit_row(make_session_state(max_new_tokens_per_unit=3), generation_step=2),
    ]

    token_ids = sample(logits, rows, special_tokens)

    assert token_ids == [
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
    repeated = make_session_state(repetition_penalty=1.5)
    repeated.generated_history.extend([50, 50])
    quiet = make_session_state(listen_prob_scale=0.5)
    rows = [
        make_unit_row(make_session_state()),
        make_unit_row(repeated),
        make_unit_row(quiet),
    ]

    token_ids = sample(logits, rows, special_tokens)

    assert token_ids == [42, 51, 60]
    assert repeated.generated_history[-1] == 51


@pytest.mark.parametrize(
    ("use_runner", "max_new_tokens", "generation_step", "closes"),
    [
        (True, 20, 0, False),
        (False, 20, 0, False),
        (False, 5, 4, True),
        (False, 6, 4, False),
    ],
)
def test_duplex_sample_masks_bad_tokens_and_closes_at_budget(
    use_runner: bool, max_new_tokens: int, generation_step: int, closes: bool
) -> None:
    """Both init paths keep the checkpoint mask; the token budget closes the chunk."""
    tokenizer = make_tokenizer()
    if use_runner:
        runner = MiniCPMOThinkerModelRunner.__new__(MiniCPMOThinkerModelRunner)
        runner.special_tokens = None
        runner.eos_token_ids = []
        special = runner.resolve_special_tokens(
            SimpleNamespace(req=Mock(tokenizer=tokenizer))
        )
    else:
        special = ThinkerAdapter(tokenizer, VOCAB_SIZE).special
    logits = torch.full((1, VOCAB_SIZE), -torch.inf)
    logits[0, 7] = 100.0
    logits[0, special.tts_pad] = 99.0
    logits[0, 42] = 0.0
    state = make_session_state(top_k=1, max_new_tokens_per_unit=max_new_tokens)

    (token_id,) = sample(logits, [make_unit_row(state, generation_step)], special)

    assert token_id == (special.chunk_eos if closes else 42)


def test_listen_mid_turn_becomes_tts_bos_and_turn_eos_ends_the_turn(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((2, VOCAB_SIZE), -10.0)
    logits[0, special_tokens.listen] = 5.0
    logits[1, special_tokens.turn_eos] = 5.0
    speaking = make_session_state()
    speaking.is_turn_ended = False
    ending = make_session_state()
    ending.is_turn_ended = False

    token_ids = sample(
        logits, [make_unit_row(speaking), make_unit_row(ending)], special_tokens
    )

    assert token_ids == [special_tokens.tts_bos, special_tokens.turn_eos]
    assert not speaking.is_turn_ended
    assert ending.is_turn_ended


def test_random_rows_sample_inside_their_top_k_and_top_p(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    torch.manual_seed(2)
    logits = torch.randn(5, VOCAB_SIZE)
    logits[:, 7] = 50.0
    # note (0xtoward): control tokens never win, so every pick comes from the second stage unchanged.
    logits[:, FIRST_SPECIAL_TOKEN_ID:] = -50.0
    temperature = 0.9
    settings = [(3, 1.0), (5, 0.6), (1, 0.8), (20, 0.95), (-1, 1.0)]
    states = [
        make_session_state(
            greedy=False, temperature=temperature, top_k=top_k, top_p=top_p
        )
        for top_k, top_p in settings
    ]
    allowed_logits = logits.clone()
    allowed_logits[
        :, build_forbidden_token_index(special_tokens, VOCAB_SIZE, logits.device)
    ] = -torch.inf

    draws = [
        sample(
            logits.clone(), [make_unit_row(state) for state in states], special_tokens
        )
        for _ in range(20)
    ]

    for row, (top_k, top_p) in enumerate(settings):
        allowed_token_ids = filter_single_row(
            allowed_logits[row] / temperature, top_k, top_p
        ).isfinite()
        assert all(
            allowed_token_ids[token_ids[row]] for token_ids in draws
        ), f"row {row} drew outside its top-k and top-p"


def test_greedy_rows_take_their_argmax_without_touching_the_random_draws(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    overflowing_temperature = 1e-38
    random_sampling = {"greedy": False, "temperature": 1.0, "top_k": 4, "top_p": 1.0}
    logits = torch.full((4, VOCAB_SIZE), -10.0)
    logits[[0, 2], 42] = 5.0
    logits[[1, 3], 50:58] = torch.linspace(4.0, 3.0, 8)
    mixed_rows = [
        make_unit_row(make_session_state(temperature=overflowing_temperature)),
        make_unit_row(make_session_state(**random_sampling)),
        make_unit_row(make_session_state(top_k=-1)),
        make_unit_row(make_session_state(**random_sampling)),
    ]
    random_rows = [
        make_unit_row(make_session_state(**random_sampling)) for _ in range(2)
    ]

    torch.manual_seed(7)
    mixed_token_ids = sample(logits.clone(), mixed_rows, special_tokens)
    torch.manual_seed(7)
    random_token_ids = sample(logits[[1, 3]].clone(), random_rows, special_tokens)

    assert mixed_token_ids[0] == 42, "a greedy row overflowing its temperature"
    assert mixed_token_ids[2] == 42, "a greedy row without a top-k"
    assert mixed_token_ids[1::2] == random_token_ids, "greedy rows changed the draws"


@pytest.mark.parametrize("temperature", [0.0, 1e-38])
def test_a_temperature_under_the_minimum_keeps_only_the_first_stage_random(
    special_tokens: MiniCPMOSpecialTokenIds, temperature: float
) -> None:
    torch.manual_seed(8)
    logits = torch.full((2, VOCAB_SIZE), -10.0)
    logits[:, special_tokens.chunk_eos] = 5.0
    logits[:, 42] = 5.0
    logits[:, 43] = 4.9

    draws = [
        sample(
            logits.clone(),
            [
                make_unit_row(
                    make_session_state(greedy=False, temperature=temperature)
                ),
                make_unit_row(make_session_state(temperature=temperature)),
            ],
            special_tokens,
        )
        for _ in range(64)
    ]

    assert {random_pick for random_pick, _ in draws} == {special_tokens.chunk_eos, 42}
    assert {greedy_pick for _, greedy_pick in draws} == {42}


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.accelerator)]
)
def test_batched_filtering_matches_the_single_row_filter_exactly(device: str) -> None:
    if device == "cuda":
        require_cuda()
    else:
        pass
    torch.manual_seed(1)
    logits = torch.randn(6, VOCAB_SIZE)
    logits[0, :4] = torch.tensor([5.0, 4.0, 4.0, 4.0])
    logits[3, :4] = torch.tensor([6.0, 3.0, 3.0, 3.0])
    # note (0xtoward): a top-p cut inside a tie keeps the lower token ids on every device.
    logits[5] = -10.0
    logits[5, 42:45] = torch.tensor([5.0, 4.0, 4.0])
    settings = [(2, 1.0), (-1, 0.5), (20, 0.9), (3, 0.5), (VOCAB_SIZE, 1.0), (2, 0.7)]
    top_k = [row_top_k for row_top_k, _ in settings]
    top_p = [row_top_p for _, row_top_p in settings]

    filtered_logits = filter_top_k_top_p(
        logits.to(device),
        top_k=torch.tensor(top_k, device=device),
        top_p=torch.tensor(top_p, device=device),
        max_top_k=max(
            (row_top_k for row_top_k in top_k if 0 < row_top_k < VOCAB_SIZE),
            default=0,
        ),
        has_top_p=any(0.0 < row_top_p < 1.0 for row_top_p in top_p),
    ).cpu()

    for row, (row_top_k, row_top_p) in enumerate(settings):
        assert torch.equal(
            filtered_logits[row], filter_single_row(logits[row], row_top_k, row_top_p)
        ), f"row {row} filtered differently"


def test_rows_ending_the_chunk_keep_their_history(
    special_tokens: MiniCPMOSpecialTokenIds,
) -> None:
    logits = torch.full((2, VOCAB_SIZE), -10.0)
    logits[0, special_tokens.chunk_eos] = 5.0
    logits[1, 42] = 5.0
    ended = make_session_state(repetition_penalty=1.5)
    ended.generated_history.extend([30, 31])
    speaking = make_session_state(repetition_penalty=1.5)

    token_ids = sample(
        logits, [make_unit_row(ended), make_unit_row(speaking)], special_tokens
    )

    assert token_ids == [special_tokens.chunk_eos, 42]
    assert ended.generated_history == [30, 31]
    assert speaking.generated_history == [42]
