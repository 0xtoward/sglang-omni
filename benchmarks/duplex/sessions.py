# SPDX-License-Identifier: Apache-2.0
"""Scripted multi-turn duplex sessions: spoken questions joined with pauses for the answers, recorded as a dataset."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
import wave
from pathlib import Path

from benchmarks.duplex.client import SAMPLE_RATE
from benchmarks.duplex.v15_dataset import Sample

RUN_KIND = "duplex-scripted-sessions"
SUBSET = "sessions"
VARIANTS = {"session": "input"}
PCM16_SAMPLE_BYTES = 2
QUESTIONS_PATH = Path(__file__).with_name("questions.txt")
DEFAULT_VOICES = ("Samantha", "Daniel", "Karen", "Alex", "Moira", "Tessa")


def session_paths(root: Path) -> list[Path]:
    return sorted(root.glob("*.wav"))


def inventory(root: Path) -> dict[str, dict[str, int] | list[str]]:
    session_count = len(session_paths(root))
    return {
        "declared": {SUBSET: session_count},
        "observed": {SUBSET: session_count},
        "ignored": [],
    }


def discover_samples(
    root: Path,
    sample_ids: list[str] | None = None,
    max_per_subset: int | None = None,
) -> list[Sample]:
    """Every session WAV under root in name order, or the ones sample_ids name."""
    paths = {f"{SUBSET}/{path.stem}": path for path in session_paths(root)}
    if sample_ids is not None:
        unknown = sorted(set(sample_ids) - paths.keys())
        if unknown:
            raise ValueError(f"unknown sessions: {unknown}")
        else:
            selected = [paths[sample_id] for sample_id in sample_ids]
    else:
        selected = list(paths.values())[:max_per_subset]
    return [
        Sample(
            id=f"{SUBSET}/{path.stem}",
            subset=SUBSET,
            directory=f"{SUBSET}/{path.stem}",
            paths={"input": path.name},
            sha256={"input": hashlib.sha256(path.read_bytes()).hexdigest()},
        )
        for path in selected
    ]


def speak_questions(questions_path: Path, voices: list[str], output_dir: Path) -> int:
    """One 16 kHz mono PCM16 WAV per line of the text file, rotating through the voices (macOS say)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        line
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    with tempfile.TemporaryDirectory() as scratch_dir:
        for index, line in enumerate(lines):
            spoken_path = Path(scratch_dir) / f"q{index}.aiff"
            subprocess.run(
                [
                    "say",
                    "-v",
                    voices[index % len(voices)],
                    "-o",
                    str(spoken_path),
                    line,
                ],
                check=True,
            )
            subprocess.run(
                [
                    "afconvert",
                    "-f",
                    "WAVE",
                    "-d",
                    f"LEI16@{SAMPLE_RATE}",
                    "-c",
                    "1",
                    str(spoken_path),
                    str(output_dir / f"q{index}.wav"),
                ],
                check=True,
            )
    return len(lines)


def build_sessions(
    questions_dir: Path,
    output_dir: Path,
    *,
    session_count: int,
    questions_per_session: int,
    lead_s: tuple[float, float],
    pause_s: tuple[float, float],
    seed: int,
) -> list[float]:
    """Each session is a random lead-in silence, then random questions, each followed by a random pause for the answer; returns the session lengths in seconds."""
    question_paths = sorted(questions_dir.glob("*.wav"))
    question_pcm = {}
    for path in question_paths:
        with wave.open(str(path), "rb") as source:
            if (
                source.getnchannels(),
                source.getsampwidth(),
                source.getframerate(),
            ) != (
                1,
                PCM16_SAMPLE_BYTES,
                SAMPLE_RATE,
            ):
                raise ValueError(f"{path} is not 16 kHz mono PCM16")
            else:
                question_pcm[path] = source.readframes(source.getnframes())
    generator = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for session in range(session_count):
        lead_seconds = generator.uniform(*lead_s)
        picks = generator.sample(question_paths, questions_per_session)
        pcm = bytes(int(lead_seconds * SAMPLE_RATE) * PCM16_SAMPLE_BYTES)
        for path in picks:
            pause_seconds = generator.uniform(*pause_s)
            pcm += question_pcm[path] + bytes(
                int(pause_seconds * SAMPLE_RATE) * PCM16_SAMPLE_BYTES
            )
        with wave.open(str(output_dir / f"s{session:03d}.wav"), "wb") as output:
            output.setparams(
                (1, PCM16_SAMPLE_BYTES, SAMPLE_RATE, 0, "NONE", "not compressed")
            )
            output.writeframes(pcm)
        manifest.append(
            {
                "session": f"s{session:03d}",
                "seconds": round(len(pcm) / PCM16_SAMPLE_BYTES / SAMPLE_RATE, 1),
                "questions": [path.name for path in picks],
                "lead_s": round(lead_seconds, 2),
            }
        )
    (output_dir / "sessions.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return [session["seconds"] for session in manifest]
