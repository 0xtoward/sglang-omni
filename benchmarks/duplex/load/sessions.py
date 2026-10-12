# SPDX-License-Identifier: Apache-2.0
"""Build multi-turn full-duplex inputs: speak questions with macOS say, then join them into sessions.

say: one 16 kHz mono PCM16 WAV per line of a text file, rotating through the given voices (macOS only).
build: each session is a random lead-in silence, then a few random questions, each followed by a random
pause long enough for the answer.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import tempfile
import wave
from pathlib import Path

SAMPLE_RATE = 16000
PCM16_SAMPLE_BYTES = 2
DEFAULT_VOICES = ("Samantha", "Daniel", "Karen", "Alex", "Moira", "Tessa")
QUESTIONS_PATH = Path(__file__).with_name("questions.txt")


def speak_questions(questions_path: Path, voices: list[str], output_dir: Path) -> None:
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
    print(f"{len(lines)} questions in {output_dir}")


def build_sessions(
    questions_dir: Path,
    output_dir: Path,
    *,
    session_count: int,
    questions_per_session: int,
    lead_s: tuple[float, float],
    pause_s: tuple[float, float],
    seed: int,
) -> None:
    question_paths = sorted(
        questions_dir.glob("q*.wav"), key=lambda path: int(path.stem[1:])
    )
    question_pcm = {}
    for path in question_paths:
        with wave.open(str(path), "rb") as source:
            question_pcm[path] = source.readframes(source.getnframes())
    generator = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for session in range(session_count):
        lead = generator.uniform(*lead_s)
        picks = generator.sample(question_paths, questions_per_session)
        pcm = bytes(int(lead * SAMPLE_RATE) * PCM16_SAMPLE_BYTES)
        for path in picks:
            pcm += question_pcm[path] + bytes(
                int(generator.uniform(*pause_s) * SAMPLE_RATE) * PCM16_SAMPLE_BYTES
            )
        with wave.open(str(output_dir / f"s{session:02d}.wav"), "wb") as output:
            output.setparams(
                (1, PCM16_SAMPLE_BYTES, SAMPLE_RATE, 0, "NONE", "not compressed")
            )
            output.writeframes(pcm)
        manifest.append(
            {
                "session": session,
                "seconds": round(len(pcm) / PCM16_SAMPLE_BYTES / SAMPLE_RATE, 1),
                "questions": [path.name for path in picks],
                "lead_s": round(lead, 2),
            }
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    seconds = [session["seconds"] for session in manifest]
    print(
        f"{session_count} sessions in {output_dir}, {min(seconds)} to {max(seconds)} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    say = commands.add_parser("say", help="speak every question line (macOS)")
    say.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    say.add_argument(
        "--voices",
        nargs="+",
        default=list(DEFAULT_VOICES),
        help="list installed voices with: say -v '?'",
    )
    say.add_argument("--output-dir", type=Path, required=True)
    build = commands.add_parser(
        "build", help="join spoken questions into multi-turn sessions"
    )
    build.add_argument("--questions-dir", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--sessions", type=int, default=32)
    build.add_argument("--questions-per-session", type=int, default=4)
    build.add_argument("--lead-s", type=float, nargs=2, default=(0.0, 6.0))
    build.add_argument("--pause-s", type=float, nargs=2, default=(10.0, 18.0))
    build.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    if args.command == "say":
        speak_questions(args.questions, args.voices, args.output_dir)
    else:
        build_sessions(
            args.questions_dir,
            args.output_dir,
            session_count=args.sessions,
            questions_per_session=args.questions_per_session,
            lead_s=tuple(args.lead_s),
            pause_s=tuple(args.pause_s),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
