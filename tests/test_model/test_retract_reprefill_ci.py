# SPDX-License-Identifier: Apache-2.0
"""Unified retraction re-prefill CI test for the projected-input-embed TTS family.

Under KV pressure the scheduler retracts a running request and re-prefills it over
``prompt + already-generated`` positions; affected models mishandle the generated
tail and either crash (qwen3_tts / moss_tts / fishaudio / zonos2) or silently
drift (higgs / voxtral). One test reproduces the whole family with a single
model-agnostic trigger and asserts with BLACK-BOX signals only — no white-box
embedding probes, which are model-specific, env-gated and easy to strip or ignore.

Trigger (unified, model-agnostic):
    admin ``POST /pause_generation {"mode":"retract"}`` retracts *every* running
    request (bypassing the "evict least-generated" policy of SGLANG_TEST_RETRACT),
    then ``/continue_generation`` forces the re-prefill. ``SGLANG_RADIX_FORCE_MISS=1``
    forces the full re-prefill (radix miss) so the generated tail is recomputed.
    Requests are long + uniform, so every retracted request carries a big tail.

Assertion 1 — liveness (always; model-agnostic, black-box):
    no fatal signature in the server log after the retract, every in-flight request
    returns HTTP 200, AND a fresh request submitted *after* the retract also returns
    200 (proves the decode stage survived, not just that in-flight streams drained).

Assertion 2 — content WER (opt-in; black-box):
    transcribe the returned audio through a served ASR model (openai/whisper-large-v3
    by default, via ``benchmarks.tasks.asr`` — the same ASR the TTS WER CI uses) and
    assert ``WER(hyp, input_text) <= budget``. Catches silent drift end-to-end with
    no model internals. Requires an intelligible generation config (a reference voice
    + temperature > 0, from the per-model preset). Skipped when the model has no WER
    budget, no reference audio, or the ASR backend is unavailable.

Per-model behaviour comes from ``RETRACT_CI_PRESETS[RETRACT_CI_MODEL]``; the weights
come from ``RETRACT_MODEL_PATH`` (a local path or an HF id).

Usage:
    RETRACT_MODEL_PATH=/models/zonos2 RETRACT_CI_MODEL=zonos2 \
    RETRACT_REF_AUDIO=/path/ref.wav RETRACT_ASR_MODEL=openai/whisper-large-v3 \
    pytest tests/test_model/test_retract_reprefill_ci.py -s -x
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pytest


# ---- per-model behaviour (weights come from RETRACT_MODEL_PATH) -------------
@dataclass(frozen=True)
class RetractPreset:
    temperature: float = 0.7
    needs_ref: bool = True
    wer_max: float | None = None  # None => content WER skipped (crash-only guard)
    lang: str = "en"


# temperature/wer_max are correctness knobs, not tuned perf numbers: temperature>0
# + a reference voice make the output intelligible enough for a stable ASR read;
# wer_max is a loose "still says the words" budget, well above the models' own
# clean-corpus WER (< 0.03), so it fails only on real retract-induced drift.
RETRACT_CI_PRESETS: dict[str, RetractPreset] = {
    "qwen3-tts": RetractPreset(temperature=0.8, needs_ref=True, wer_max=0.15),
    "moss": RetractPreset(temperature=0.9, needs_ref=True, wer_max=0.20),
    "higgs": RetractPreset(temperature=0.7, needs_ref=False, wer_max=0.15),
    "fish": RetractPreset(temperature=0.7, needs_ref=True, wer_max=0.20),
    "voxtral": RetractPreset(temperature=0.7, needs_ref=False, wer_max=0.20),
    "zonos2": RetractPreset(temperature=1.15, needs_ref=True, wer_max=0.20),
}

MODEL_PATH = os.environ.get("RETRACT_MODEL_PATH")
CI_MODEL = os.environ.get("RETRACT_CI_MODEL")
REF_AUDIO = os.environ.get("RETRACT_REF_AUDIO")
ASR_MODEL = os.environ.get("RETRACT_ASR_MODEL", "openai/whisper-large-v3")
REUSE_ASR_PORT = os.environ.get("RETRACT_ASR_PORT")  # reuse an already-running ASR
RUN_WER = os.environ.get("RETRACT_RUN_WER", "1") != "0"

PORT = int(os.environ.get("RETRACT_PORT", "18011"))
ASR_LAUNCH_PORT = int(os.environ.get("RETRACT_ASR_LAUNCH_PORT", str(PORT + 1)))
MAX_TOTAL_TOKENS = int(os.environ.get("RETRACT_MAX_TOTAL_TOKENS", "8192"))
STARTUP_TIMEOUT = int(os.environ.get("RETRACT_STARTUP_TIMEOUT", "600"))
N_REQUESTS = int(os.environ.get("RETRACT_N_REQUESTS", "4"))
MAX_NEW_TOKENS = int(os.environ.get("RETRACT_MAX_NEW_TOKENS", "1500"))
RETRACT_WAIT_S = float(os.environ.get("RETRACT_WAIT_S", "6"))

BASE = f"http://127.0.0.1:{PORT}"

# Distinct clean passages (known WER targets), each long enough that every request
# still carries a big generated tail at RETRACT_WAIT_S.
_TEXTS = [
    "In the quiet hours before dawn the old lighthouse keeper climbed the spiral "
    "stair, counting each worn step as the sea wind pressed against the glass, and "
    "by morning the lamp still burned steady across the calm and waking water.",
    "The market opened early that Saturday, and the vendors arranged bright crates "
    "of oranges and pears while the baker carried warm loaves to the corner stall "
    "and children chased the pigeons across the wet grey stones of the square.",
    "She folded the last letter carefully, pressed it beneath the heavy book, and "
    "watched the rain trace slow rivers down the window as the kettle began to sing "
    "and the room filled with the ordinary comfort of a long and quiet evening.",
    "Far beyond the harbor the fishing boats turned for home, their lanterns "
    "swinging against the dark, and the whole village listened for the familiar "
    "creak of timber and rope that meant the men had come safely back again.",
]

# Signatures that must NOT appear in the server log after the retract.
_FATAL = re.compile(
    r"shape of the mask|cannot be broadcast|is invalid for input|Size mismatch|"
    r"Dead stage process|RuntimeError|Traceback \(most recent",
    re.I,
)


def _preset() -> RetractPreset:
    if CI_MODEL and CI_MODEL in RETRACT_CI_PRESETS:
        return RETRACT_CI_PRESETS[CI_MODEL]
    # Unknown/unset model name -> crash-only guard; use a ref if one was provided.
    return RetractPreset(needs_ref=bool(REF_AUDIO), wer_max=None)


# ---- server plumbing --------------------------------------------------------
def _launch(model_path: str, port: int, log_path: Path, *, extra_env=None, extra_args=()):
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [
            "sgl-omni", "serve", "--model-path", model_path,
            "--host", "127.0.0.1", "--port", str(port), *extra_args,
        ],
        stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(extra_env or {})},
    )
    return proc, log


def _wait_healthy(proc: subprocess.Popen, port: int, log_path: Path) -> None:
    import requests

    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if proc.poll() is not None:  # process died => startup failure (no regex guessing)
            tail = Path(log_path).read_text(errors="replace")[-2000:]
            raise RuntimeError(f"server on :{port} exited rc={proc.returncode}\n{tail}")
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"server on :{port} not healthy in {STARTUP_TIMEOUT}s")


def _stop(proc: subprocess.Popen, log) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    try:
        log.close()
    except Exception:  # noqa: BLE001
        pass


# ---- request driving --------------------------------------------------------
async def _upload_voice(session: aiohttp.ClientSession) -> bool:
    if not REF_AUDIO:
        return False
    form = aiohttp.FormData()
    form.add_field("name", "retractci")
    form.add_field("consent", "retract-ci")
    form.add_field("ref_text", "This is a reference voice for the retraction test.")
    form.add_field(
        "audio_sample", Path(REF_AUDIO).read_bytes(),
        filename=Path(REF_AUDIO).name, content_type="audio/wav",
    )
    async with session.post(f"{BASE}/v1/audio/voices", data=form) as r:
        return r.status == 200


async def _speech(session, idx, text, preset, voice_ok, wav_dir):
    payload = {
        "input": text,
        "response_format": "wav",
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": preset.temperature,
    }
    if voice_ok:
        payload["voice"] = "retractci"
    elif REF_AUDIO:
        payload["ref_audio"] = REF_AUDIO
    async with session.post(f"{BASE}/v1/audio/speech", json=payload) as r:
        body = await r.read()
        if r.status == 200:
            path = wav_dir / f"req_{idx}.wav"
            path.write_bytes(body)
            return idx, 200, str(path)
        return idx, r.status, body[:300].decode("utf-8", "replace")


async def _admin(session, path, payload):
    async with session.post(f"{BASE}/{path}", json=payload) as r:
        return r.status


async def _drive(preset: RetractPreset, wav_dir: Path):
    """Fire long requests, retract-all mid-generation, continue, collect WAVs."""
    texts = [_TEXTS[i % len(_TEXTS)] for i in range(N_REQUESTS)]
    timeout = aiohttp.ClientTimeout(total=900)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        voice_ok = await _upload_voice(s) if preset.needs_ref else False
        tasks = [
            asyncio.create_task(_speech(s, i, texts[i], preset, voice_ok, wav_dir))
            for i in range(N_REQUESTS)
        ]
        await asyncio.sleep(RETRACT_WAIT_S)  # let every request build a generated tail
        await _admin(s, "pause_generation", {"mode": "retract"})  # retract ALL running
        await asyncio.sleep(1.0)
        await _admin(s, "continue_generation", {})  # -> re-prefill the generated tail
        results = await asyncio.gather(*tasks)
        # A fresh request AFTER the retract: proves the decode stage still serves,
        # not merely that the in-flight streams drained.
        fresh = await _speech(s, N_REQUESTS, texts[0], preset, voice_ok, wav_dir)
    return results, texts, fresh


# ---- content WER (opt-in, black-box) ---------------------------------------
def _assert_black_box_wer(wav_dir: Path, texts: list[str], preset: RetractPreset) -> None:
    try:
        from benchmarks.metrics.wer import SampleOutput
        from benchmarks.tasks.asr import load_router_asr, transcribe_and_compute_wer
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"benchmarks ASR layer unavailable, skipping WER: {exc}")

    asr_proc = asr_log = None
    if REUSE_ASR_PORT:
        asr_port = int(REUSE_ASR_PORT)
    else:
        try:
            from tests.utils import wait_for_gpu_memory_release

            wait_for_gpu_memory_release()
        except Exception:  # noqa: BLE001
            time.sleep(5)  # TTS server already stopped; give the driver time to free VRAM
        asr_port = ASR_LAUNCH_PORT
        asr_log = wav_dir.parent / "asr_server.log"
        asr_proc, asr_log_f = _launch(ASR_MODEL, asr_port, asr_log)
        _wait_healthy(asr_proc, asr_port, asr_log)

    try:
        asr = load_router_asr(asr_port, model_path=ASR_MODEL)
        wers: list[tuple[int, float]] = []
        for i, text in enumerate(texts):
            wav = wav_dir / f"req_{i}.wav"
            if not wav.exists():
                continue
            out = SampleOutput(sample_id=f"req_{i}", target_text=text)
            out = transcribe_and_compute_wer(out, str(wav), asr, preset.lang, "cuda:0")
            assert out.is_success, f"ASR failed on req_{i}: {out.error}"
            wers.append((i, out.wer))
        assert wers, "no generated WAVs to transcribe"
        worst_i, worst = max(wers, key=lambda x: x[1])
        print(f"\n[RETRACT WER] per-request={[(i, round(w, 3)) for i, w in wers]}")
        assert worst <= preset.wer_max, (
            f"retract re-prefill drifted: req_{worst_i} WER={worst:.3f} > "
            f"budget {preset.wer_max} (all={[(i, round(w, 3)) for i, w in wers]})"
        )
    finally:
        if asr_proc is not None:
            _stop(asr_proc, asr_log_f)


# ---- the test ---------------------------------------------------------------
def test_retraction_reprefill_survives(tmp_path):
    if not MODEL_PATH:
        pytest.skip("set RETRACT_MODEL_PATH to run the retraction re-prefill test")

    preset = _preset()
    tts_log = tmp_path / "tts_server.log"
    wav_dir = tmp_path / "wavs"
    wav_dir.mkdir()
    marker = f"===RETRACT_CI_{int(time.time())}==="

    proc, log = _launch(
        MODEL_PATH, PORT, tts_log,
        extra_env={"SGLANG_RADIX_FORCE_MISS": "1"},
        extra_args=["--max-total-tokens", str(MAX_TOTAL_TOKENS)],
    )
    try:
        _wait_healthy(proc, PORT, tts_log)
        with open(tts_log, "a") as f:
            f.write(marker + "\n")

        results, texts, fresh = asyncio.run(_drive(preset, wav_dir))

        # --- Assertion 1: liveness (black-box, model-agnostic) ---
        after = Path(tts_log).read_text(errors="replace").split(marker, 1)[-1]
        fatal = _FATAL.search(after)
        assert not fatal, (
            f"server hit a fatal error on retraction re-prefill: {fatal.group(0)}\n"
            f"{after[-1500:]}"
        )
        bad = [(i, st, d) for (i, st, d) in results if st != 200]
        assert not bad, f"in-flight requests failed after retraction re-prefill: {bad}"
        assert fresh[1] == 200, (
            f"the decode stage did not survive: post-retract request -> "
            f"{fresh[1]} {fresh[2]}"
        )
    finally:
        _stop(proc, log)

    # --- Assertion 2: content WER (opt-in, black-box) ---
    if preset.wer_max is None:
        return  # crash-only model / unknown preset: liveness is the whole guard
    if not RUN_WER:
        return
    if preset.needs_ref and not REF_AUDIO:
        pytest.skip("set RETRACT_REF_AUDIO for the black-box WER check on this model")
    _assert_black_box_wer(wav_dir, texts, preset)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
