# SPDX-License-Identifier: Apache-2.0
"""Unified retraction re-prefill CI test for projected-input-embed TTS models.

One model-agnostic test that reproduces the whole bug family: under KV pressure
the scheduler retracts a running request and re-prefills it over
``prompt + already-generated`` positions. Affected models mishandle the generated
tail and either crash (qwen3_tts / moss_tts / zonos2 / fishaudio) or silently
drift (higgs / voxtral).

Trigger is model-agnostic and deterministic: instead of racing a real KV-OOM, we
drive the admin endpoint ``POST /pause_generation {"mode":"retract"}`` which
retracts *every* running request (bypassing the "evict least-generated" policy of
``SGLANG_TEST_RETRACT``), then ``/continue_generation`` to force the re-prefill.
``SGLANG_RADIX_FORCE_MISS=1`` forces the full re-prefill (radix miss) so the
generated tail is actually recomputed. Requests are long + uniform so every
retracted request carries a substantial generated tail.

Assertions (crash family): the server stays alive (no dead-stage / shape-mismatch
/ RuntimeError in the log) and every request completes HTTP 200.

Silent-content models (higgs / voxtral) additionally set ``*_REPREFILL_DIAG=1`` if
the model runner ships the env-gated acceptance probe, and the test asserts the
probe reports ``dist_to_fused ~ 0`` on the re-prefilled generated positions.

Usage:
    RETRACT_MODEL_PATH=/path/to/model pytest tests/test_model/test_retract_reprefill_ci.py -s -x
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from pathlib import Path

import aiohttp
import pytest

MODEL_PATH = os.environ.get("RETRACT_MODEL_PATH")
REF_AUDIO = os.environ.get("RETRACT_REF_AUDIO")  # wav path for voice-clone models
PORT = int(os.environ.get("RETRACT_PORT", "18011"))
MAX_TOTAL_TOKENS = int(os.environ.get("RETRACT_MAX_TOTAL_TOKENS", "8192"))
STARTUP_TIMEOUT = int(os.environ.get("RETRACT_STARTUP_TIMEOUT", "600"))
N_REQUESTS = int(os.environ.get("RETRACT_N_REQUESTS", "3"))
MAX_NEW_TOKENS = int(os.environ.get("RETRACT_MAX_NEW_TOKENS", "1500"))

BASE = f"http://127.0.0.1:{PORT}"
_LONG = (
    "A long paragraph repeated many times to force long uniform generation so "
    "every running request carries a big already-generated tail at the moment we "
    "retract them all and re-prefill the generated portion from a radix-missed "
    "prompt, which is the condition that exposes the retraction re-prefill bug. "
) * 4

# Fatal signatures that must NOT appear after the retract.
_FATAL = re.compile(
    r"shape of the mask|cannot be broadcast|Size mismatch|batch_size|"
    r"Dead stage process|RuntimeError|Traceback \(most recent",
    re.I,
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    if not MODEL_PATH:
        pytest.skip("set RETRACT_MODEL_PATH to run the retraction re-prefill test")
    log_path = tmp_path_factory.mktemp("retract") / "server.log"
    env = {
        **os.environ,
        "SGLANG_RADIX_FORCE_MISS": "1",  # force full re-prefill of the generated tail
        "HIGGS_REPREFILL_DIAG": "1",
        "FISH_REPREFILL_DIAG": "1",
    }
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [
                "sgl-omni", "serve", "--model-path", MODEL_PATH,
                "--max-total-tokens", str(MAX_TOTAL_TOKENS),
                "--host", "127.0.0.1", "--port", str(PORT),
            ],
            stdout=log, stderr=subprocess.STDOUT, env=env,
        )
    try:
        _wait_listening(log_path)
        yield log_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def _wait_listening(log_path: Path) -> None:
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            import socket

            with socket.create_connection(("127.0.0.1", PORT), timeout=2):
                return
        except OSError:
            if log_path.exists() and _FATAL.search(log_path.read_text(errors="replace")):
                raise RuntimeError(f"server failed to start:\n{log_path.read_text()[-2000:]}")
            time.sleep(3)
    raise TimeoutError("server did not start in time")


async def _upload_voice(session, name):
    if not REF_AUDIO:
        return False
    form = aiohttp.FormData()
    form.add_field("name", name)
    form.add_field("consent", "retract-ci")
    form.add_field("ref_text", "This is a reference voice for the retraction test.")
    form.add_field(
        "audio_sample", Path(REF_AUDIO).read_bytes(),
        filename=Path(REF_AUDIO).name, content_type="audio/wav",
    )
    async with session.post(f"{BASE}/v1/audio/voices", data=form) as r:
        return r.status == 200


async def _speech(session, idx, voice_ok):
    payload = {
        "input": f"Item {idx}. {_LONG}",
        "response_format": "wav",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": 0.0,
    }
    if voice_ok:
        payload["voice"] = "retractci"
    elif REF_AUDIO:
        payload["ref_audio"] = REF_AUDIO
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        body = await r.read()
        return r.status, len(body), body[:300].decode("utf-8", "replace")


async def _admin(session, path, payload):
    async with session.post(f"{BASE}/{path}", json=payload) as r:
        return r.status


async def _drive():
    """Fire long requests, retract-all mid-generation, continue, collect results."""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
        voice_ok = await _upload_voice(s, "retractci")
        tasks = [asyncio.create_task(_speech(s, i, voice_ok)) for i in range(N_REQUESTS)]
        await asyncio.sleep(12)  # let every request build a generated tail
        await _admin(s, "pause_generation", {"mode": "retract"})  # retract ALL running
        await asyncio.sleep(1)
        await _admin(s, "continue_generation", {})  # -> re-prefill the generated tail
        return await asyncio.gather(*tasks)


def test_retraction_reprefill_survives(server):
    log_path = server
    marker = f"===RETRACT_CI_{int(time.time())}==="
    with open(log_path, "a") as f:
        f.write(marker + "\n")

    results = asyncio.run(_drive())

    # crash family: no fatal signature after the retract, every request 200.
    after = log_path.read_text(errors="replace").split(marker, 1)[-1]
    fatal = _FATAL.search(after)
    assert not fatal, f"server hit a fatal error on retraction re-prefill: {fatal.group(0)}\n{after[-1500:]}"

    bad = [(st, detail) for st, _, detail in results if st != 200]
    assert not bad, f"requests failed after retraction re-prefill: {bad}"

    # silent-content models: if the acceptance probe ran, the re-prefilled generated
    # positions must match the fused target (dist_to_fused ~ 0). Absent probe -> skip.
    probe = re.findall(r"REPREFILL_DIAG .*?dist_to_fused=([0-9.]+)", after)
    if probe:
        worst = max(float(x) for x in probe)
        assert worst < 1e-3, f"generated tail not correctly rebuilt: max dist_to_fused={worst}"
