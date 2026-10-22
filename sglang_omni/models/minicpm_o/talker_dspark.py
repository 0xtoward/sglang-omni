# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding for the MiniCPM-o talker, with verification that keeps its sampling exact.

The draft is a DSpark checkpoint over the talker's codec ids that reads the talker's
residual stream after a few of its layers. The talker samples with its own shaping
(codec suppress mask, a repetition penalty over the last 16 outputs, and the stop id
masked until the minimum length) before temperature, top-k and top-p, so the worker
draws every proposal from the draft's logits shaped the same way and rejection-samples
it against the talker's distribution for each verify row (talker_dspark_sampling.py).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from sglang.kernels.ops.speculative.dspark.dspark_accept import FinalizeAcceptLens
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.dspark import Qwen3DSparkModel, StepSampler, VanillaMarkov
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult
from sglang.srt.speculative.dspark_components.dspark_verify import (
    AcceptOuts,
    TargetVerifyExecutor,
)
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.utils import DynamicGradMode
from torch import nn
from transformers import Qwen3Config

from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.models.minicpm_o.talker_dspark_sampling import (
    TalkerSamplingState,
    exact_accept,
    sample_draft_block,
)
from sglang_omni.models.minicpm_o.talker_model_runner import REP_PENALTY_WINDOW
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

DRAFT_ARCHITECTURE = "MiniCPMOTalkerDSparkDraftModel"
# Fields SGLang reads from the top level of a DSpark config; SpecForge exports nest them.
LIFTED_DSPARK_FIELDS = (
    "markov_rank",
    "markov_head_type",
    "mask_token_id",
    "target_layer_ids",
)


def talker_vocabulary_weights(model_path: str) -> dict[str, torch.Tensor]:
    """The talker's codec embedding and weight-normed codec head from its checkpoint."""
    with open(os.path.join(model_path, "model.safetensors.index.json")) as index_file:
        weight_map = json.load(index_file)["weight_map"]

    def tensor(name: str) -> torch.Tensor:
        with safe_open(os.path.join(model_path, weight_map[name]), "pt") as shard:
            return shard.get_tensor(name)

    gain = tensor("tts.head_code.0.parametrizations.weight.original0").float()
    direction = tensor("tts.head_code.0.parametrizations.weight.original1").float()
    return {
        "embed_tokens.weight": tensor("tts.emb_code.0.weight").contiguous(),
        "lm_head.weight": (gain * direction / direction.norm(dim=1, keepdim=True))
        .to(torch.bfloat16)
        .contiguous(),
    }


def materialize_draft_checkpoint(
    draft_model_path: str, model_path: str, revision: str | None
) -> tuple[str, int]:
    """Write a loadable copy of a SpecForge DSpark draft for the talker.

    The copy names the talker draft class, lifts the DSpark fields SGLang reads to the
    top level, turns off the confidence head (every verify checks a whole block), links
    the draft weights and adds the talker's codec embedding and head, which the draft
    owns. Returns the copy's path and the draft's block size.
    """
    source_dir = (
        draft_model_path
        if os.path.isdir(draft_model_path)
        else snapshot_download(draft_model_path, revision=revision)
    )
    with open(os.path.join(source_dir, "config.json")) as config_file:
        config = json.load(config_file)
    nested = config.get("dflash_config") or {}
    for field_name in LIFTED_DSPARK_FIELDS:
        if field_name in nested and field_name not in config:
            config[field_name] = nested[field_name]
        else:
            pass
    if int(config.get("markov_rank", 0)) <= 0:
        raise ValueError(f"{draft_model_path} is not a DSpark draft (no Markov head)")
    elif config.get("markov_head_type", "vanilla") != "vanilla":
        raise ValueError("the MiniCPM-o talker draft supports the vanilla Markov head")
    else:
        pass
    vocabulary = talker_vocabulary_weights(model_path)
    vocab_size = int(vocabulary["lm_head.weight"].shape[0])
    if int(config["vocab_size"]) != vocab_size:
        raise ValueError(
            f"{draft_model_path} has {config['vocab_size']} ids, the talker {vocab_size}"
        )
    else:
        pass
    config.update(
        architectures=[DRAFT_ARCHITECTURE],
        enable_confidence_head=False,
        tie_word_embeddings=False,
    )
    nested["enable_confidence_head"] = False
    config["dflash_config"] = nested
    draft_dir = tempfile.mkdtemp(prefix="minicpm-o-talker-dspark-")
    with open(os.path.join(draft_dir, "config.json"), "w") as config_file:
        json.dump(config, config_file, indent=2)
    for file_name in os.listdir(source_dir):
        if file_name.startswith("model") and file_name.endswith(".safetensors"):
            os.symlink(
                os.path.realpath(os.path.join(source_dir, file_name)),
                os.path.join(draft_dir, file_name),
            )
        else:
            pass
    save_file(vocabulary, os.path.join(draft_dir, "talker_vocabulary.safetensors"))
    return draft_dir, int(config["block_size"])


BlockDraw = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, TalkerSamplingState],
    tuple[torch.Tensor, torch.Tensor],
]
BlockAccept = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, TalkerSamplingState],
    tuple[torch.Tensor, torch.Tensor],
]


def draw_block(
    base_logits: torch.Tensor,
    previous_tokens: torch.Tensor,
    markov_embedding: torch.Tensor,
    markov_projection: torch.Tensor,
    state: TalkerSamplingState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """sample_draft_block with its Exp(1) noise drawn here."""
    return sample_draft_block(
        base_logits,
        previous_tokens,
        markov_embedding,
        markov_projection,
        state,
        torch.empty_like(base_logits, dtype=torch.float32).exponential_(),
    )


def accept_block(
    target_logits: torch.Tensor,
    shaped_draft_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    state: TalkerSamplingState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """exact_accept with its uniform and Exp(1) noise drawn here."""
    return exact_accept(
        target_logits,
        shaped_draft_logits,
        draft_tokens,
        state,
        torch.rand(draft_tokens.shape, device=draft_tokens.device),
        torch.empty(
            (draft_tokens.shape[0], target_logits.shape[1]), device=draft_tokens.device
        ).exponential_(),
    )


class SpeculativeSamplingGraphs:
    """The draft draw and the exact verify replayed as device graphs per padded batch size.

    Eager, each is a compiled-function dispatch plus dozens of small kernels per step,
    several times the Talker's GPU work for a block; a replay launches them at once.
    Batches above the largest size, or whose state was not loaded, take the fallbacks.
    """

    def __init__(
        self,
        *,
        batch_sizes: list[int],
        gamma: int,
        vocab: int,
        eos_id: int,
        markov_head: VanillaMarkov,
        draw_fallback: BlockDraw,
        accept_fallback: BlockAccept,
        backend: DeviceGraphBackend,
        device: torch.device,
    ) -> None:
        self.batch_sizes = sorted(set(batch_sizes))
        self.draw_fallback = draw_fallback
        self.accept_fallback = accept_fallback
        self.markov_head = markov_head
        self.rows_per_request = gamma + 1
        largest = self.batch_sizes[-1]
        self.state = TalkerSamplingState(
            recent_tokens=torch.full(
                (largest, REP_PENALTY_WINDOW), vocab, dtype=torch.long, device=device
            ),
            generated=torch.zeros(largest, dtype=torch.long, device=device),
            min_new_tokens=torch.zeros(largest, dtype=torch.long, device=device),
            penalties=torch.ones(largest, device=device),
            suppress=torch.zeros((largest, vocab), dtype=torch.bool, device=device),
            temperatures=torch.ones(largest, device=device),
            top_ks=torch.full((largest,), vocab, dtype=torch.long, device=device),
            top_ps=torch.ones(largest, device=device),
            greedy=torch.zeros(largest, dtype=torch.bool, device=device),
            eos_id=eos_id,
        )
        self.base_logits = torch.zeros((largest, gamma, vocab), device=device)
        self.previous_tokens = torch.zeros(largest, dtype=torch.long, device=device)
        self.target_logits = torch.zeros(
            (largest * self.rows_per_request, vocab), device=device
        )
        self.shaped_draft_logits = torch.zeros((largest, gamma, vocab), device=device)
        self.draft_tokens = torch.zeros(
            (largest, gamma), dtype=torch.long, device=device
        )
        self.drawn_tokens = torch.zeros(
            (largest, gamma), dtype=torch.long, device=device
        )
        self.drawn_logits = torch.zeros((largest, gamma, vocab), device=device)
        self.accepted_count = torch.zeros(largest, dtype=torch.long, device=device)
        self.next_tokens = torch.zeros(largest, dtype=torch.long, device=device)
        self.loaded_count = 0
        self.draw_graphs: dict[int, ReplayableGraph] = {}
        self.accept_graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        pool = backend.graph_pool_handle()
        stream = device_module.Stream(device=device)
        # note (0xtoward): largest first on one stream, so smaller graphs reuse its pool.
        for size in reversed(self.batch_sizes):
            stream.wait_stream(device_module.current_stream(device))
            with device_module.stream(stream):
                for _ in range(2):
                    self.run_draw(size)
                    self.run_accept(size)
            # note (0xtoward): code2wav runs on other threads of this process, so a
            # capture failure there must not abort this one.
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.run_draw(size)
            self.draw_graphs[size] = graph
            with backend.capture(
                pool=pool, stream=stream, thread_local_errors=True
            ) as graph:
                self.run_accept(size)
            self.accept_graphs[size] = graph
            device_module.current_stream(device).wait_stream(stream)

    def state_rows(self, size: int) -> TalkerSamplingState:
        state = self.state
        return TalkerSamplingState(
            recent_tokens=state.recent_tokens[:size],
            generated=state.generated[:size],
            min_new_tokens=state.min_new_tokens[:size],
            penalties=state.penalties[:size],
            suppress=state.suppress[:size],
            temperatures=state.temperatures[:size],
            top_ks=state.top_ks[:size],
            top_ps=state.top_ps[:size],
            greedy=state.greedy[:size],
            eos_id=state.eos_id,
        )

    def run_draw(self, size: int) -> None:
        tokens, logits = draw_block(
            self.base_logits[:size],
            self.previous_tokens[:size],
            self.markov_head.markov_w1.weight,
            self.markov_head.markov_w2.weight,
            self.state_rows(size),
        )
        self.drawn_tokens[:size].copy_(tokens)
        self.drawn_logits[:size].copy_(logits)

    def run_accept(self, size: int) -> None:
        accepted_count, next_tokens = accept_block(
            self.target_logits[: size * self.rows_per_request],
            self.shaped_draft_logits[:size],
            self.draft_tokens[:size],
            self.state_rows(size),
        )
        self.accepted_count[:size].copy_(accepted_count)
        self.next_tokens[:size].copy_(next_tokens)

    def padded_size(self, count: int) -> int | None:
        return next((size for size in self.batch_sizes if size >= count), None)

    def load(self, state: TalkerSamplingState) -> None:
        """Copy one step's sampling state into the graphs' inputs; rows past it stay valid."""
        count = int(state.generated.shape[0])
        if self.padded_size(count) is None:
            self.loaded_count = 0
            return
        else:
            pass
        for name in (
            "recent_tokens",
            "generated",
            "min_new_tokens",
            "penalties",
            "suppress",
            "temperatures",
            "top_ks",
            "top_ps",
            "greedy",
        ):
            getattr(self.state, name)[:count].copy_(getattr(state, name))
        self.loaded_count = count

    def draw(
        self,
        base_logits: torch.Tensor,
        previous_tokens: torch.Tensor,
        markov_embedding: torch.Tensor,
        markov_projection: torch.Tensor,
        state: TalkerSamplingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = base_logits.shape[0]
        if count != self.loaded_count:
            return self.draw_fallback(
                base_logits, previous_tokens, markov_embedding, markov_projection, state
            )
        else:
            pass
        self.base_logits[:count].copy_(base_logits)
        self.previous_tokens[:count].copy_(previous_tokens)
        self.draw_graphs[self.padded_size(count)].replay()
        return self.drawn_tokens[:count].clone(), self.drawn_logits[:count].clone()

    def accept(
        self,
        target_logits: torch.Tensor,
        shaped_draft_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        state: TalkerSamplingState,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = draft_tokens.shape[0]
        if count != self.loaded_count:
            return self.accept_fallback(
                target_logits, shaped_draft_logits, draft_tokens, state
            )
        else:
            pass
        self.target_logits[: count * self.rows_per_request].copy_(target_logits)
        self.shaped_draft_logits[:count].copy_(shaped_draft_logits)
        self.draft_tokens[:count].copy_(draft_tokens)
        self.accept_graphs[self.padded_size(count)].replay()
        return self.accepted_count[:count].clone(), self.next_tokens[:count].clone()


class MirroredMarkovHead(VanillaMarkov):
    """Markov head that draws each draft step from the talker's shaping of it.

    Rejection sampling divides by the proposal distribution, so the returned corrected
    logits are exactly the shaped logits the draft drew from.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.sampling_state: TalkerSamplingState | None = None
        self.block_sampler: BlockDraw = draw_block

    def sample_block(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_tokens: torch.Tensor,
        hidden_states: torch.Tensor | None,
        sampler: StepSampler,
        collect_corrected: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Draw the block from the talker-shaped logits.

        sampler, hidden_states and collect_corrected go unused: the given step sampler
        would skip the shaping, and the shaped logits are always returned.
        """
        state = self.sampling_state
        assert state is not None, "a draft step needs its sampling state"
        return self.block_sampler(
            base_logits,
            first_prev_tokens.long(),
            self.markov_w1.weight,
            self.markov_w2.weight,
            state,
        )


class MiniCPMOTalkerDSparkDraftModel(Qwen3DSparkModel):
    """DSpark draft over the talker's codec ids, with the talker's embedding and head."""

    uses_own_vocab_modules: bool = True

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        vocab_size = int(config.vocab_size)
        hidden_size = int(config.hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.markov_head = MirroredMarkovHead(
            vocab_size=vocab_size, markov_rank=int(config.markov_rank)
        )

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return (hidden @ self.lm_head.weight.T).float(), None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        vocabulary: dict[str, torch.Tensor] = {}
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            normalized_name = name.removeprefix("model.")
            if normalized_name in ("embed_tokens.weight", "lm_head.weight"):
                vocabulary[normalized_name] = loaded_weight
            else:
                backbone_weights.append((name, loaded_weight))
        super().load_weights(backbone_weights)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(vocabulary["embed_tokens.weight"])
            self.lm_head.weight.copy_(vocabulary["lm_head.weight"])


def build_sampling_state(
    batch: ScheduleBatch, sampling_info: SamplingBatchInfo, eos_id: int
) -> TalkerSamplingState:
    """Each running request's committed history and the talker's sampling parameters."""
    device = sampling_info.temperatures.device
    vocab = int(sampling_info.vocab_size)
    recent: list[list[int]] = []
    request_values: list[list[float]] = []
    suppress = torch.zeros((len(batch.reqs), vocab), dtype=torch.bool)
    for row, request in enumerate(batch.reqs):
        tail = [int(token_id) for token_id in request.output_ids[-REP_PENALTY_WINDOW:]]
        recent.append([vocab] * (REP_PENALTY_WINDOW - len(tail)) + tail)
        inputs = request.omni_data.talker_model_inputs
        request_values.append(
            [
                float(inputs.get("rep_penalty", 1.0)),
                len(request.output_ids),
                int(inputs.get("min_new_tokens", 0)),
            ]
        )
        suppress_tokens = request.omni_data.suppress_tokens or getattr(
            request,
            "_codec_suppress_tokens",  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            None,
        )
        for token_id in suppress_tokens or ():
            if 0 <= int(token_id) < vocab:
                suppress[row, int(token_id)] = True
            else:
                pass
    # One host-to-device copy carries the three per-request values.
    values = torch.tensor(request_values, dtype=torch.float64).to(device)
    top_ks = sampling_info.top_ks.view(-1)
    top_ps = sampling_info.top_ps.view(-1).float()
    return TalkerSamplingState(
        recent_tokens=torch.tensor(recent, dtype=torch.long).to(device),
        generated=values[:, 1].long(),
        min_new_tokens=values[:, 2].long(),
        penalties=values[:, 0].float(),
        suppress=suppress.to(device),
        temperatures=sampling_info.temperatures.view(-1).float(),
        top_ks=top_ks,
        top_ps=top_ps,
        greedy=top_ks == 1,
        eos_id=eos_id,
    )


class ExactVerifyExecutor(TargetVerifyExecutor):
    """Accept drafts against the talker's exact per-position distribution."""

    sampling_state: TalkerSamplingState | None = None
    accept_function: BlockAccept = staticmethod(accept_block)

    def accept_and_finalize(
        self,
        *,
        folded_accept: bool,
        bs: int,
        verify_ids_2d: torch.Tensor,
        target_logits: torch.Tensor | None,
        draft_block: DraftBlockResult,
        sampling_info: SamplingBatchInfo,
        draft_input: DFlashDraftInputV2,
        layout: None,
        prefix_lens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> AcceptOuts:
        state = self.sampling_state
        assert not folded_accept and layout is None and target_logits is not None
        assert state is not None, "a verify step needs its sampling state"
        draft_tokens = draft_tokens.long()
        accepted_count, next_tokens = self.accept_function(
            target_logits, draft_block.corrected_logits, draft_tokens, state
        )
        finalized = FinalizeAcceptLens.execute(
            correct_len=accepted_count,
            cap_trim_lens=torch.zeros_like(accepted_count),
            prefix_lens=prefix_lens,
        )
        out_tokens = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=accepted_count,
            bonus=next_tokens,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            gamma=self.gamma,
        )
        return AcceptOuts(
            correct_len=accepted_count,
            bonus=next_tokens,
            cap_trim_lens=finalized.cap_trim_lens,
            commit_lens=finalized.commit_lens,
            new_seq_lens=finalized.new_seq_lens,
            out_tokens=out_tokens,
        )


class MiniCPMOTalkerDSparkWorker(DSparkWorkerV2):
    """DSpark worker that mirrors the talker's sampling in drafting and verification."""

    sampling_graphs: SpeculativeSamplingGraphs | None = None

    def __init__(
        self,
        *,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: SpeculativeTargetWorker,
    ) -> None:
        # note (0xtoward): the draft draw and the acceptance noise are local, so
        # tensor-parallel ranks would disagree on the committed tokens.
        if get_parallel().tp_size != 1:
            raise ValueError("MiniCPM-o talker speculative decoding runs on one GPU")
        elif (
            read_ragged_verify_mode() != RaggedVerifyMode.STATIC
            or envs.SGLANG_SIMULATE_ACC_LEN.get() > 0
        ):
            raise ValueError(
                "MiniCPM-o talker speculative decoding verifies whole blocks; unset "
                "SGLANG_RAGGED_VERIFY_MODE and SGLANG_SIMULATE_ACC_LEN"
            )
        else:
            pass
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # The stage builder always materializes the draft under this class.
        assert isinstance(self.draft_model, MiniCPMOTalkerDSparkDraftModel)
        self.eos_id = int(self.target_worker.model_runner.model.codec_eos_id)
        self.exact_verify_executor = ExactVerifyExecutor(
            target_worker=self.target_worker,
            gamma=self.gamma,
            verify_num_draft_tokens=self.verify_num_draft_tokens,
            model_runner=self.model_runner,
            kv_injector=self._kv_injector,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            tp_sync=self._tp_sync,  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self._verify_executor = (
            self.exact_verify_executor
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        markov_head = self.draft_model.markov_head
        # note (0xtoward): batches the graphs do not cover draw and verify compiled; eager,
        # each is about a hundred small kernels per step.
        self.compiled_draw = torch.compile(draw_block, dynamic=True)
        self.compiled_accept = torch.compile(accept_block, dynamic=True)
        markov_head.block_sampler = self.compiled_draw
        self.exact_verify_executor.accept_function = self.compiled_accept
        self.warm_up_compiled_sampling()

    def init_cuda_graphs(self) -> None:
        """Capture the sampling graphs too, once the request slots they are sized by exist."""
        super().init_cuda_graphs()
        device = torch.device(self.device)
        backend = current_platform.get_device_graph_backend(device)
        if backend is None:
            return
        else:
            pass
        markov_head = self.draft_model.markov_head
        slots = self.target_worker.model_runner.req_to_token_pool.req_to_token.shape[0]
        self.sampling_graphs = SpeculativeSamplingGraphs(
            batch_sizes=[size for size in (1, 2, 4, 8, 16, 32) if size <= slots] or [1],
            gamma=self.gamma,
            vocab=int(self.draft_model.lm_head.weight.shape[0]),
            eos_id=self.eos_id,
            markov_head=markov_head,
            draw_fallback=self.compiled_draw,
            accept_fallback=self.compiled_accept,
            backend=backend,
            device=device,
        )
        markov_head.block_sampler = self.sampling_graphs.draw
        self.exact_verify_executor.accept_function = self.sampling_graphs.accept

    # note (0xtoward): compiled graphs guard on grad mode, so the warm-up runs under the
    # scheduler loop's mode.
    @DynamicGradMode()
    def warm_up_compiled_sampling(self) -> None:
        """Compile the draft block and the exact verify before the first request.

        Dynamic shapes still specialize a single running request, so both one and two
        requests are traced here instead of on the serving thread.
        """
        markov_head = self.draft_model.markov_head
        vocab = int(self.draft_model.lm_head.weight.shape[0])
        for batch_size in (1, 2):
            zeros = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            ones = torch.ones(batch_size, dtype=torch.float32, device=self.device)
            state = TalkerSamplingState(
                recent_tokens=torch.full(
                    (batch_size, REP_PENALTY_WINDOW),
                    vocab,
                    dtype=torch.long,
                    device=self.device,
                ),
                generated=zeros,
                min_new_tokens=zeros,
                penalties=ones,
                suppress=torch.zeros(
                    (batch_size, vocab), dtype=torch.bool, device=self.device
                ),
                temperatures=ones,
                top_ks=torch.full(
                    (batch_size,), vocab, dtype=torch.int32, device=self.device
                ),
                top_ps=ones,
                greedy=torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                eos_id=self.eos_id,
            )
            base_logits = torch.zeros(
                (batch_size, self.gamma, vocab), device=self.device
            )
            draft_tokens, shaped_logits = markov_head.block_sampler(
                base_logits,
                zeros,
                markov_head.markov_w1.weight,
                markov_head.markov_w2.weight,
                state,
            )
            self.exact_verify_executor.accept_function(
                torch.zeros((batch_size * (self.gamma + 1), vocab), device=self.device),
                shaped_logits,
                draft_tokens,
                state,
            )

    def _maybe_build_draft_sampler(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self, *, available_memory_gb: float
    ) -> None:
        """The folded in-graph sampler would skip the mirrored Markov head."""
        return None

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: None = None,
        grammar_barrier: None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        is_decode = not (
            batch.forward_mode.is_extend()
            or batch.is_extend_in_batch
            or batch.forward_mode.is_idle()
        )
        if is_decode:
            sampling_info = batch.sampling_info
            # The talker's request builder sets none of these, and the exact verify has
            # no term for them.
            assert not (
                sampling_info.need_min_p_sampling
                or sampling_info.logit_bias is not None
                or sampling_info.has_custom_logit_processor
                or batch.return_logprob
            )
            state = build_sampling_state(batch, sampling_info, self.eos_id)
            self.draft_model.markov_head.sampling_state = state
            self.exact_verify_executor.sampling_state = state
            if self.sampling_graphs is not None:
                self.sampling_graphs.load(state)
            else:
                pass
        else:
            pass
        return super().forward_batch_generation(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )
