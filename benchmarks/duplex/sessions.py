# SPDX-License-Identifier: Apache-2.0
"""Scripted multi-turn duplex sessions: spoken questions joined with pauses for the answers, recorded as a dataset.

A session WAV may come with <stem>.frames.json, its own camera frames ({unit: JPEG path relative to the root}), as the
video sessions built from Video-MME rows do.
"""

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
    return [sample_for(path) for path in selected]


def sample_for(path: Path) -> Sample:
    files = {"input": path}
    frames_path = path.with_name(f"{path.stem}.frames.json")
    if frames_path.exists():
        files["frames"] = frames_path
    else:
        pass
    return Sample(
        id=f"{SUBSET}/{path.stem}",
        subset=SUBSET,
        directory=f"{SUBSET}/{path.stem}",
        paths={key: file.name for key, file in files.items()},
        sha256={
            key: hashlib.sha256(file.read_bytes()).hexdigest()
            for key, file in files.items()
        },
    )


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


def spoken_pcm(text: str, voice: str) -> bytes:
    """One line spoken by macOS say, as 16 kHz mono PCM16."""
    with tempfile.TemporaryDirectory() as scratch_dir:
        spoken_path = Path(scratch_dir) / "line.aiff"
        wav_path = Path(scratch_dir) / "line.wav"
        subprocess.run(["say", "-v", voice, "-o", str(spoken_path), text], check=True)
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
                str(wav_path),
            ],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as source:
            return source.readframes(source.getnframes())


def video_question_text(row: dict[str, object]) -> str:
    options = " ".join(
        f"Option {option[0]}: {option[3:].rstrip('.')}." for option in row["options"]
    )
    return (
        f"{row['question']} {options} Which option is correct? Answer with the letter."
    )


def build_video_sessions(
    rows_path: Path,
    videos_root: Path,
    output_dir: Path,
    *,
    voice: str,
    frame_every_units: int,
    max_video_s: float,
    answer_pause_s: float,
) -> list[float]:
    """One session per Video-MME row: the video's soundtrack with a frame every frame_every_units units, then the spoken question and a pause for the answer.

    Frames are 1 fps JPEGs under frames/<video>/; returns the session lengths in seconds.
    """
    rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for row in rows:
        video_path = videos_root / row["video_path"]
        video = video_path.stem
        frames_dir = output_dir / "frames" / video
        if not frames_dir.exists():
            frames_dir.mkdir(parents=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-t",
                    str(max_video_s),
                    "-i",
                    str(video_path),
                    "-vf",
                    "fps=1,scale=640:-2",
                    "-q:v",
                    "5",
                    str(frames_dir / "f%05d.jpg"),
                ],
                check=True,
            )
        else:
            pass
        soundtrack = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-t",
                str(max_video_s),
                "-i",
                str(video_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "s16le",
                "-",
            ],
            check=True,
            capture_output=True,
        ).stdout
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        unit_bytes = SAMPLE_RATE * PCM16_SAMPLE_BYTES
        video_units = min(len(frame_paths), len(soundtrack) // unit_bytes)
        question_start_s = float(video_units)
        pcm = (
            soundtrack[: video_units * unit_bytes]
            + spoken_pcm(video_question_text(row), voice)
            + bytes(int(answer_pause_s * SAMPLE_RATE) * PCM16_SAMPLE_BYTES)
        )
        name = f"v{row['question_id']}"
        with wave.open(str(output_dir / f"{name}.wav"), "wb") as output:
            output.setparams(
                (1, PCM16_SAMPLE_BYTES, SAMPLE_RATE, 0, "NONE", "not compressed")
            )
            output.writeframes(pcm)
        (output_dir / f"{name}.frames.json").write_text(
            json.dumps(
                {
                    str(unit): str(frame_paths[unit].relative_to(output_dir))
                    for unit in range(0, video_units, frame_every_units)
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        manifest.append(
            {
                "session": name,
                "seconds": round(len(pcm) / PCM16_SAMPLE_BYTES / SAMPLE_RATE, 1),
                "video": video,
                "question_id": row["question_id"],
                "question_start_s": question_start_s,
                "answer": row["answer"],
                "task_type": row.get("task_type"),
            }
        )
    (output_dir / "sessions.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return [session["seconds"] for session in manifest]
