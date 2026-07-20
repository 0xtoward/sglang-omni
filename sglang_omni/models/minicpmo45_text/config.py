"""Declarative minimal MiniCPM-o 4.5 text pipeline."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import PipelineConfig, PlacementConfig, StageConfig

_PKG = "sglang_omni.models.minicpmo45_text"


def _stages() -> list[StageConfig]:
    return [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"max_seq_len": 2048},
            runtime_arg_map={"max_seq_len": "max_seq_len"},
            next="thinker",
        ),
        StageConfig(
            name="thinker",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_thinker_executor_from_config",
            factory_args={"thinker_max_seq_len": 2048},
            runtime_arg_map={"max_seq_len": "thinker_max_seq_len"},
            gpu=0,
            next="decode",
        ),
        StageConfig(
            name="decode",
            process="pipeline",
            factory=f"{_PKG}.stages.create_decode_executor",
            terminal=True,
        ),
    ]


class MiniCPMO45TextPipelineConfig(PipelineConfig):
    """One NPU, one thinker, text-only MiniCPM-o 4.5 smoke pipeline."""

    architecture: ClassVar[str] = "MiniCPMO"

    model_path: str
    name: str = "minicpmo45-text-ascend-minimal"
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )
    stages: list[StageConfig] = Field(default_factory=_stages)


EntryClass = MiniCPMO45TextPipelineConfig
