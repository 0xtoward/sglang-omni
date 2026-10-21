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
from collections.abc import Iterable

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


class MirroredMarkovHead(VanillaMarkov):
    """Markov head that draws each draft step from the talker's shaping of it.

    Rejection sampling divides by the proposal distribution, so the returned corrected
    logits are exactly the shaped logits the draft drew from.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.sampling_state: TalkerSamplingState | None = None
        self.block_sampler = sample_draft_block

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
            torch.empty_like(base_logits, dtype=torch.float32).exponential_(),
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
    accept_function = staticmethod(exact_accept)

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
            target_logits,
            draft_block.corrected_logits,
            draft_tokens,
            state,
            torch.rand(draft_tokens.shape, device=draft_tokens.device),
            torch.empty(
                (bs, target_logits.shape[1]), device=draft_tokens.device
            ).exponential_(),
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
        # note (0xtoward): eager, the draft block and the exact verify are about a hundred
        # small kernels per step; compiled they fuse into a few.
        self.exact_verify_executor.accept_function = torch.compile(
            exact_accept, dynamic=True
        )
        self.draft_model.markov_head.block_sampler = torch.compile(
            sample_draft_block, dynamic=True
        )
        self.warm_up_compiled_sampling()

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
                torch.ones_like(base_logits),
            )
            self.exact_verify_executor.accept_function(
                torch.zeros((batch_size * (self.gamma + 1), vocab), device=self.device),
                shaped_logits,
                draft_tokens,
                state,
                torch.zeros((batch_size, self.gamma), device=self.device),
                torch.ones((batch_size, vocab), device=self.device),
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
        else:
            pass
        return super().forward_batch_generation(
            batch,
            on_publish,
            grammar_barrier,
            pp_proxy_tensors=pp_proxy_tensors,
        )
