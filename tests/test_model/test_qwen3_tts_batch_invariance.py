# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS deterministic inference checks."""

from __future__ import annotations

import asyncio
import io
import os
import wave
from collections.abc import Iterator
from pathlib import Path

import aiohttp
import pytest
import torch
import yaml

from benchmarks.benchmarker.utils import managed_omni_server
from benchmarks.dataset.prepare import DATASETS, download_dataset
from benchmarks.dataset.seedtts import load_seedtts_samples
from tests.test_model.omni_router_utils import _find_available_port_range
from tests.utils import server_log_file

MODEL_PATH = os.environ.get(
    "QWEN3_TTS_TEST_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
)
DATASET = os.environ.get("QWEN3_TTS_TEST_DATASET", DATASETS["seedtts-50"])
SEED = 123456
STARTUP_TIMEOUT = 600


def _pcm(wav_bytes: bytes) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        pcm = wav.readframes(wav.getnframes())
    assert pcm
    return pcm


async def _model_info(base_url: str, stop: asyncio.Event) -> int:
    max_batch_size = 0
    timeout = aiohttp.ClientTimeout(total=2)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not stop.is_set():
            try:
                async with session.get(f"{base_url}/model_info") as response:
                    response.raise_for_status()
                    payload = await response.json()
            except TimeoutError:
                pass
            else:
                for result in payload.get("results", []):
                    if result.get("stage") == "tts_engine":
                        max_batch_size = max(
                            max_batch_size,
                            int(result.get("data", {}).get("running_batch_size", 0)),
                        )
            await asyncio.sleep(0.05)
    return max_batch_size


async def _generate(
    session: aiohttp.ClientSession,
    base_url: str,
    payload: dict,
) -> bytes:
    async with session.post(f"{base_url}/v1/audio/speech", json=payload) as response:
        response.raise_for_status()
        return _pcm(await response.read())


async def _run_batch_invariance_check(
    base_url: str,
    serial_payloads: list[dict],
    batch_payloads: list[dict],
) -> tuple[list[bytes], list[bytes]]:
    stop = asyncio.Event()
    poller = asyncio.create_task(_model_info(base_url, stop))
    timeout = aiohttp.ClientTimeout(total=300)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            b1 = [
                await _generate(session, base_url, payload)
                for payload in serial_payloads
            ]
            b8 = await asyncio.gather(
                *(_generate(session, base_url, payload) for payload in batch_payloads)
            )
    finally:
        stop.set()
        max_batch_size = await poller

    assert max_batch_size == 8
    return b1, b8


def _payload(sample) -> dict:
    return {
        "model": MODEL_PATH,
        "input": sample.target_text,
        "ref_audio": sample.ref_audio,
        "ref_text": sample.ref_text,
        "response_format": "wav",
        "seed": SEED,
        "max_new_tokens": 2048,
    }


@pytest.fixture(scope="module")
def qwen3_tts_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[str]:
    if not torch.cuda.is_available():
        pytest.skip("Qwen3-TTS batch invariance requires CUDA")
    if not Path(DATASET).exists():
        download_dataset(DATASET, quiet=True)
    config_path = tmp_path_factory.mktemp("qwen3_tts_deterministic") / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "config_cls": "Qwen3TTSPipelineConfig",
                "model_path": MODEL_PATH,
                "enable_deterministic_inference": True,
                "runtime_overrides": {
                    "tts_engine": {
                        "server_args_overrides": {
                            "attention_backend": "triton",
                            "max_running_requests": 8,
                            "cuda_graph_max_bs": 8,
                            "cuda_graph_bs": [1, 2, 4, 8],
                            "torch_compile_max_bs": 8,
                        }
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    port = _find_available_port_range(1)
    log_file = server_log_file(tmp_path_factory, "qwen3_tts_deterministic")
    with managed_omni_server(
        model_path=MODEL_PATH,
        port=port,
        host="127.0.0.1",
        log_file=log_file,
        server_config=str(config_path),
        timeout=STARTUP_TIMEOUT,
    ):
        yield f"http://127.0.0.1:{port}"


@pytest.mark.benchmark
def test_qwen3_tts_deterministic_b1_matches_b8(qwen3_tts_server: str) -> None:
    """Match one request across physical Talker batch sizes one and eight."""
    sample = load_seedtts_samples(DATASET, 1, split="en")[0]
    payload = _payload(sample)
    b1, b8 = asyncio.run(
        _run_batch_invariance_check(
            qwen3_tts_server,
            [payload] * 3,
            [payload] * 8,
        )
    )
    assert len(set(b1)) == 1
    assert all(pcm == b1[0] for pcm in b8)


@pytest.mark.benchmark
def test_qwen3_tts_deterministic_mixed_b8_matches_b1(
    qwen3_tts_server: str,
) -> None:
    """Match each row in a mixed batch to its batch-one result."""
    payloads = []
    for sample in load_seedtts_samples(DATASET, 8, split="en"):
        payload = _payload(sample)
        payload["input"] = " ".join([sample.target_text] * 3)
        payloads.append(payload)
    b1, b8 = asyncio.run(
        _run_batch_invariance_check(
            qwen3_tts_server,
            payloads,
            payloads,
        )
    )
    assert b8 == b1
