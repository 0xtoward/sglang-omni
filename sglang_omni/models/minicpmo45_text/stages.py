"""Factories for the MiniCPM-o 4.5 Ascend text-only smoke pipeline."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoTokenizer

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    create_streaming_detokenize_scheduler,
)
from sglang_omni.models.qwen3_omni.payload_types import Qwen3OmniPipelineState
from sglang_omni.models.qwen3_omni.request_builders import (
    make_thinker_scheduler_adapters,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler


def _tokenize_text_request(tokenizer: Any, payload: Any) -> tuple[torch.Tensor, str]:
    """Turn the OpenAI bridge's text inputs into a MiniCPM prompt tensor."""

    inputs = payload.request.inputs
    if isinstance(inputs, list) and all(isinstance(token, int) for token in inputs):
        return torch.tensor(inputs, dtype=torch.long), ""

    if isinstance(inputs, str):
        messages: list[dict[str, Any]] = [{"role": "user", "content": inputs}]
    elif isinstance(inputs, dict):
        messages = inputs.get("messages", [])
    else:
        messages = inputs
    if not isinstance(messages, list):
        raise TypeError("MiniCPM-o minimal pipeline expects text messages")

    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
    except (AttributeError, TypeError, ValueError):
        text = "\n".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict)
        )
        token_ids = tokenizer.encode(text, add_special_tokens=True)
    return torch.as_tensor(token_ids, dtype=torch.long).flatten(), ""


def create_preprocessing_executor(model_path: str, *, max_seq_len: int = 2048):
    """CPU-side tokenizer stage; deliberately excludes image/audio handling."""

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _preprocess(payload):
        input_ids, prompt_text = _tokenize_text_request(tokenizer, payload)
        max_new_tokens = int((payload.request.params or {}).get("max_new_tokens", 32))
        if input_ids.numel() + max_new_tokens >= max_seq_len:
            raise ValueError(
                f"Requested {input_ids.numel()} prompt + {max_new_tokens} output tokens "
                f"exceeds minimal pipeline context {max_seq_len}."
            )
        payload.data = Qwen3OmniPipelineState(
            prompt={
                "prompt_text": prompt_text,
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            },
            stream_state={"token_ids": [], "text": ""},
        ).to_dict()
        return payload

    return SimpleScheduler(_preprocess)


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = 2048,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Create a real SGLang core MiniCPMO worker on the active platform."""

    overrides: dict[str, Any] = {
        "device": "npu",
        "dtype": "bfloat16",
        "disable_cuda_graph": True,
        "disable_overlap_schedule": True,
        "sampling_backend": "pytorch",
        "max_running_requests": 1,
        "mem_fraction_static": 0.72,
        "tp_size": tp_size,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        model_arch_override="MiniCPMO",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    request_builder, result_adapter = make_thinker_scheduler_adapters(
        tokenizer=tokenizer,
        vocab_size=model_config.vocab_size,
        thinker_config=None,
    )
    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=ModelRunner(model_worker, SGLangOutputProcessor()),
        request_builder=request_builder,
        result_adapter=result_adapter,
    )


def create_decode_executor(model_path: str):
    return create_streaming_detokenize_scheduler(model_path)
