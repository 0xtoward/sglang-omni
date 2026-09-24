from __future__ import annotations

import asyncio

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_sglang_thinker_request
from sglang_omni.models.minicpm_o.talker_request import build_talker_request
from sglang_omni.proto import OmniRequest, StagePayload

# The generation suffix from MiniCPM-o-4_5's tokenizer template.
GENERATION_TEMPLATE = """
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
    {%- if use_tts_template is defined and use_tts_template is true %}
        {{- '<|tts_bos|>' }}
    {%- endif %}
{%- endif %}
"""


@pytest.mark.parametrize("use_tts_template", [False, True])
def test_chat_prompt_matches_checkpoint_non_thinking_default(
    use_tts_template: bool,
) -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = PreTrainedTokenizerBase(chat_template=GENERATION_TEMPLATE)

    prompt = preprocessor.render_chat_template(
        [{"role": "user", "content": "Answer the question."}],
        use_tts_template=use_tts_template,
    )

    expected = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if use_tts_template:
        expected += "<|tts_bos|>"
    assert prompt == expected


def test_raw_prompt_bypasses_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    raw_prompt = "<|im_start|>assistant\n<think>\n"

    assert preprocessor.render_chat_template(raw_prompt) == raw_prompt


def test_prompt_token_ids_bypass_chat_template() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    token_ids = [151644, 151667, 198]
    payload = StagePayload(
        request_id="prompt-token-ids",
        request=OmniRequest(inputs={"messages": token_ids}),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))

    assert result.data["prompt"]["prompt_text"] == ""
    assert result.data["prompt"]["input_ids"].tolist() == token_ids
    torch.testing.assert_close(
        result.data["prompt"]["attention_mask"], torch.ones(3, dtype=torch.long)
    )


class KnownTextTokenizer:
    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|tts_bos|>": 151703, "<|tts_eos|>": 151704}[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert text == "你好"
        assert not add_special_tokens
        return [100, 101]


def test_known_tts_text_prefills_the_speech_span() -> None:
    preprocessor = object.__new__(MiniCPMOPreprocessor)
    preprocessor.tokenizer = KnownTextTokenizer()
    preprocessor.speech_enabled = True
    payload = StagePayload(
        request_id="known-text",
        request=OmniRequest(
            inputs={"messages": [151703]},
            params={"known_tts_text": "你好"},
            metadata={"output_modalities": ["audio"]},
        ),
        data=None,
    )

    result = asyncio.run(preprocessor(payload))
    prompt = result.data["prompt"]
    assert prompt["input_ids"].tolist() == [151703, 100, 101, 151704]
    assert prompt["known_tts_output_ids"] == [100, 101]

    state = MiniCPMOPipelineState.from_dict(result.data)
    state.thinker_out = {
        "output_ids": [999],
        "extra_model_outputs": {
            "hidden_states_seq": [torch.full((4,), i) for i in range(4)]
        },
    }
    span = build_talker_request(
        state,
        tts_bos_token_id=151703,
        tts_eos_token_id=151704,
    )
    assert span["tts_token_ids"].tolist() == [100, 101]
    torch.testing.assert_close(
        span["tts_hidden"],
        torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]]),
    )


def test_known_tts_text_keeps_its_prefill_out_of_prefix_cache() -> None:
    state = MiniCPMOPipelineState(
        prompt={
            "prompt_text": "",
            "input_ids": torch.tensor([151703, 100, 101, 151704]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "known_tts_output_ids": [100, 101],
        }
    )
    first = build_sglang_thinker_request(
        state,
        params={},
        tokenizer=KnownTextTokenizer(),
        vocab_size=151808,
        request_id="first",
    ).req
    second = build_sglang_thinker_request(
        state,
        params={},
        tokenizer=KnownTextTokenizer(),
        vocab_size=151808,
        request_id="second",
    ).req

    assert first.extra_key != second.extra_key
    assert first.skip_radix_cache_insert
    assert second.skip_radix_cache_insert
