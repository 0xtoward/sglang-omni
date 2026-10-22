# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o talker speculative sampling against the talker's own sampling procedure."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.srt.models.dspark import VanillaMarkov
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from torch import nn

from sglang_omni.models.minicpm_o.talker_dspark import (
    ExactAcceptGraphs,
    FoldedTalkerDraw,
    StaticSamplingRows,
    accept_block,
    draw_block,
)
from sglang_omni.models.minicpm_o.talker_dspark_sampling import (
    TalkerSamplingState,
    exact_accept,
    nucleus,
    sample_draft_block,
    shape_logits,
    target_probabilities,
    window_counts,
)
from sglang_omni.models.minicpm_o.talker_model_runner import (
    REP_PENALTY_WINDOW,
    TalkerSlotState,
)
from sglang_omni.platforms import current_platform
from tests.unit_test.fixtures.accelerator import require_cuda

VOCAB_SIZE = 12
CODEC_EOS_ID = VOCAB_SIZE - 1


def recent_tokens(outputs: list[int]) -> torch.Tensor:
    tail = outputs[-REP_PENALTY_WINDOW:]
    return torch.tensor(
        [VOCAB_SIZE] * (REP_PENALTY_WINDOW - len(tail)) + tail, dtype=torch.long
    )


def make_state(histories: list[list[int]], **overrides: object) -> TalkerSamplingState:
    batch = len(histories)
    values: dict[str, object] = dict(
        recent_tokens=torch.stack([recent_tokens(history) for history in histories]),
        generated=torch.tensor([len(history) for history in histories]),
        min_new_tokens=torch.full((batch,), 6),
        penalties=torch.full((batch,), 1.3),
        suppress=torch.zeros(batch, VOCAB_SIZE, dtype=torch.bool),
        temperatures=torch.full((batch,), 0.8),
        top_ks=torch.full((batch,), 6),
        top_ps=torch.full((batch,), 0.9),
        greedy=torch.zeros(batch, dtype=torch.bool),
        eos_id=CODEC_EOS_ID,
    )
    values.update(overrides)
    return TalkerSamplingState(**values)


def slot_state_shaping(
    logits: torch.Tensor, histories: list[list[int]], state: TalkerSamplingState
) -> torch.Tensor:
    """The runner's own shaping: TalkerSlotState after appending each history token."""
    slots = TalkerSlotState.allocate(
        len(histories), VOCAB_SIZE, CODEC_EOS_ID, torch.device("cpu")
    )
    rows = torch.arange(len(histories))
    slots.min_new_tokens[rows] = state.min_new_tokens
    slots.penalties[rows] = state.penalties
    slots.suppress[rows] = state.suppress
    for row, history in enumerate(histories):
        for token in history:
            slots.append(torch.tensor([row]), torch.tensor([token]))
    shaped = logits.clone()
    slots.apply(shaped, rows)
    return shaped


def test_shaping_matches_the_talker_slot_state() -> None:
    torch.manual_seed(0)
    histories = [
        torch.randint(0, CODEC_EOS_ID, (length,)).tolist() for length in (0, 2, 16, 23)
    ]
    suppress = torch.zeros(len(histories), VOCAB_SIZE, dtype=torch.bool)
    suppress[1, 4] = True
    state = make_state(histories, suppress=suppress)
    logits = torch.randn(len(histories), VOCAB_SIZE) * 3

    shaped = shape_logits(
        logits,
        suppress=suppress,
        counts=window_counts(state.recent_tokens, VOCAB_SIZE),
        penalties=state.penalties,
        eos_masked=state.generated < state.min_new_tokens,
        eos_id=CODEC_EOS_ID,
    )

    assert torch.equal(shaped, slot_state_shaping(logits, histories, state))


def test_verify_rows_count_the_drafts_before_them() -> None:
    torch.manual_seed(1)
    histories = [
        torch.randint(0, CODEC_EOS_ID, (length,)).tolist() for length in (1, 15, 30)
    ]
    drafts = torch.randint(0, VOCAB_SIZE, (len(histories), 4))
    state = make_state(histories)
    logits = torch.randn(len(histories) * 5, VOCAB_SIZE) * 2

    target = target_probabilities(logits, drafts, state)

    for row, history in enumerate(histories):
        for offset in range(5):
            outputs = history + drafts[row, :offset].tolist()
            one = make_state([outputs])
            shaped = slot_state_shaping(
                logits[row * 5 + offset : row * 5 + offset + 1], [outputs], one
            )
            expected = nucleus(
                shaped,
                temperatures=one.temperatures,
                top_ks=one.top_ks,
                top_ps=one.top_ps,
                greedy=one.greedy,
            )
            torch.testing.assert_close(target[row, offset], expected[0])


def run_speculative_step(
    trials: int, history: list[int], greedy: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One draft block and verify per trial; returns committed tokens and the target logits."""
    drafts = 3
    state = make_state(
        [history] * trials, greedy=torch.full((trials,), greedy, dtype=torch.bool)
    )
    # Target logits depend on the row and on the token before it, as a model's would.
    position_logits = torch.randn(drafts + 1, VOCAB_SIZE) * 2
    token_logits = torch.randn(VOCAB_SIZE, VOCAB_SIZE)
    draft_base = torch.randn(drafts, VOCAB_SIZE) * 2
    markov_embedding = torch.randn(VOCAB_SIZE, 4) * 0.3
    markov_projection = torch.randn(VOCAB_SIZE, 4) * 0.3
    tokens, shaped = sample_draft_block(
        draft_base.unsqueeze(0).expand(trials, -1, -1),
        torch.full((trials,), history[-1]),
        markov_embedding,
        markov_projection,
        state,
        torch.empty(trials, drafts, VOCAB_SIZE).exponential_(),
    )
    previous = torch.cat([torch.full((trials, 1), history[-1]), tokens], dim=1)
    accepted, bonus = exact_accept(
        (position_logits.unsqueeze(0) + token_logits[previous]).reshape(-1, VOCAB_SIZE),
        shaped,
        tokens,
        state,
        torch.rand(trials, drafts),
        torch.empty(trials, VOCAB_SIZE).exponential_(),
    )
    committed = torch.cat([tokens, torch.zeros(trials, 1, dtype=torch.long)], dim=1)
    committed.scatter_(1, accepted.unsqueeze(1), bonus.unsqueeze(1))
    return committed, accepted, position_logits, token_logits


def talker_distribution(
    position_logits: torch.Tensor,
    token_logits: torch.Tensor,
    outputs: list[int],
    offset: int,
    greedy: bool = False,
) -> torch.Tensor:
    """The talker's next-token distribution after outputs, sampled one step at a time."""
    state = make_state([outputs], greedy=torch.tensor([greedy]))
    logits = (position_logits[offset] + token_logits[outputs[-1]]).unsqueeze(0)
    return target_probabilities(logits, torch.zeros(1, 0, dtype=torch.long), state)[
        0, 0
    ]


def test_speculative_step_reproduces_talker_sampling() -> None:
    torch.manual_seed(2)
    trials, history = 400_000, [3, 5, 3, 7]
    committed, accepted, position_logits, token_logits = run_speculative_step(
        trials, history, greedy=False
    )

    first = committed[:, 0]
    first_target = talker_distribution(position_logits, token_logits, history, 0)
    first_observed = torch.bincount(first, minlength=VOCAB_SIZE).float() / trials
    first_sigma = (first_target * (1 - first_target) / trials).sqrt().clamp_min(1e-9)
    assert ((first_observed - first_target).abs() / first_sigma).max() < 4.5

    # Once the likeliest token is committed as an accepted draft, the step's second
    # token, accepted or drawn, follows the talker's distribution after it.
    likeliest = int(first_target.argmax())
    picked = (accepted >= 1) & (first == likeliest)
    second_target = talker_distribution(
        position_logits, token_logits, history + [likeliest], 1
    )
    second_observed = (
        torch.bincount(committed[picked, 1], minlength=VOCAB_SIZE).float()
        / picked.sum()
    )
    second_sigma = (
        (second_target * (1 - second_target) / picked.sum()).sqrt().clamp_min(1e-9)
    )
    assert ((second_observed - second_target).abs() / second_sigma).max() < 4.5


def test_greedy_rows_commit_the_talker_argmax() -> None:
    torch.manual_seed(3)
    history = [2, 9, 2]
    committed, accepted, position_logits, token_logits = run_speculative_step(
        64, history, greedy=True
    )

    outputs = list(history)
    for offset in range(int(accepted[0]) + 1):
        expected = talker_distribution(
            position_logits, token_logits, outputs, offset, greedy=True
        ).argmax()
        assert torch.all(committed[:, offset] == expected)
        outputs.append(int(expected))


class TinyDraft(nn.Module):
    """The draft model surface the folded draw reads: head, Markov head, block layout."""

    sample_from_anchor = False

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.lm_head = nn.Linear(hidden, VOCAB_SIZE, bias=False)
        self.markov_head = VanillaMarkov(vocab_size=VOCAB_SIZE, markov_rank=4)

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden @ self.lm_head.weight.T).float(), None


def test_folded_draw_hook_draws_the_talker_shaped_block() -> None:
    torch.manual_seed(5)
    gamma, hidden = 3, 8
    model = TinyDraft(hidden)
    rows = StaticSamplingRows(
        capacity=4, vocab=VOCAB_SIZE, eos_id=CODEC_EOS_ID, device=torch.device("cpu")
    )
    state = make_state([[3, 5, 3], [1], [2, 2, 2, 2]])
    rows.load(state)
    hook = FoldedTalkerDraw(
        model=model,
        gamma=gamma,
        max_bs=4,
        device=torch.device("cpu"),
        tp_sync=SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0)),
        rows=rows,
        draw_kernel=sample_draft_block,
    )
    # The draft graph feeds the anchor and gamma mask positions of each request.
    hidden_states = torch.randn(3 * (gamma + 1), hidden)
    input_ids = torch.tensor([[3, 0, 0, 0], [1, 0, 0, 0], [2, 0, 0, 0]]).view(-1)

    torch.manual_seed(6)
    with torch.no_grad():
        hook(hidden_states, input_ids)
    torch.manual_seed(6)
    with torch.no_grad():
        base_logits = model.compute_base_logits(
            hidden_states.view(3, gamma + 1, hidden)[:, 1:].reshape(-1, hidden)
        )[0].view(3, gamma, -1)
        expected_tokens, expected_shaped = sample_draft_block(
            base_logits,
            torch.tensor([3, 1, 2]),
            model.markov_head.markov_w1.weight,
            model.markov_head.markov_w2.weight,
            state,
            torch.empty_like(base_logits).exponential_(),
        )
    assert torch.equal(hook.out[: 3 * gamma].view(3, gamma), expected_tokens)
    torch.testing.assert_close(
        hook.corrected_out[: 3 * gamma].view(3, gamma, -1), expected_shaped
    )


def test_accept_graphs_replay_the_greedy_verify_on_the_folded_block() -> None:
    require_cuda()
    device = torch.device("cuda")
    torch.manual_seed(4)
    gamma = 3
    markov_head = VanillaMarkov(vocab_size=VOCAB_SIZE, markov_rank=4).to(device)
    rows = StaticSamplingRows(
        capacity=4, vocab=VOCAB_SIZE, eos_id=CODEC_EOS_ID, device=device
    )
    draft_tokens_buffer = torch.zeros(4 * gamma, dtype=torch.long, device=device)
    shaped_buffer = torch.zeros(4 * gamma, VOCAB_SIZE, device=device)
    graphs = ExactAcceptGraphs(
        batch_sizes=[1, 4],
        gamma=gamma,
        rows=rows,
        draft_tokens=draft_tokens_buffer,
        shaped_draft_logits=shaped_buffer,
        fallback=accept_block,
        backend=current_platform.get_device_graph_backend(device),
        device=device,
    )
    cpu_state = make_state(
        [[3, 5, 3], [1], [2, 2, 2, 2]], greedy=torch.ones(3, dtype=torch.bool)
    )
    state = TalkerSamplingState(
        **{
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in vars(cpu_state).items()
        }
    )
    base_logits = torch.randn(3, gamma, VOCAB_SIZE, device=device)
    previous = torch.tensor([3, 1, 2], device=device)
    target_logits = torch.randn(3 * (gamma + 1), VOCAB_SIZE, device=device)
    weights = (markov_head.markov_w1.weight, markov_head.markov_w2.weight)
    tokens, shaped = draw_block(base_logits, previous, *weights, state)

    # A folded draw leaves its block in the buffers the graphs read.
    rows.load(state)
    draft_tokens_buffer[: 3 * gamma].copy_(tokens.reshape(-1))
    shaped_buffer[: 3 * gamma].copy_(shaped.reshape(3 * gamma, -1))
    accepted, bonus = graphs.accept(
        target_logits,
        shaped_buffer[: 3 * gamma].view(3, gamma, -1),
        draft_tokens_buffer[: 3 * gamma].view(3, gamma),
        state,
    )
    expected_accepted, expected_bonus = accept_block(
        target_logits, shaped, tokens, state
    )
    assert torch.equal(accepted, expected_accepted)
    assert torch.equal(bonus, expected_bonus)

    # An eager draw's block is copied in first.
    accepted, bonus = graphs.accept(
        target_logits, shaped.clone(), tokens.clone(), state
    )
    assert torch.equal(accepted, expected_accepted)
    assert torch.equal(bonus, expected_bonus)
