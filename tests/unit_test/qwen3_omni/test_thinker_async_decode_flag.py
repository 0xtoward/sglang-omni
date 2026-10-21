# SPDX-License-Identifier: Apache-2.0
"""The public async flag reaches the factory without losing omitted vs explicit values."""
from __future__ import annotations

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig


@pytest.mark.parametrize("enable_async_decode", [None, "false", "true"])
def test_dotted_flag_reaches_the_thinker_factory(
    enable_async_decode: str | None,
) -> None:
    """An omitted flag must leave the factory free to resolve the draft-specific default."""
    cfg = Qwen3OmniPipelineConfig(model_path="dummy")
    overrides = [
        ("thinker.engine.speculative_algorithm", "DFLASH"),
        ("thinker.engine.speculative_draft_model_path", "draft"),
    ]
    if enable_async_decode is not None:
        overrides.append(("thinker.factory.enable_async_decode", enable_async_decode))
    else:
        pass
    resolved = ConfigManager(cfg).merge_config(overrides)
    factory_kwargs = resolve_stage_typed_kwargs(resolved.stage_named("thinker"))

    if enable_async_decode is None:
        assert "enable_async_decode" not in factory_kwargs
    else:
        assert factory_kwargs["enable_async_decode"] is (enable_async_decode == "true")
    assert factory_kwargs["server_args_overrides"] == {
        "speculative_algorithm": "DFLASH",
        "speculative_draft_model_path": "draft",
    }
