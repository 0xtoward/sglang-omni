# SPDX-License-Identifier: Apache-2.0
"""Build scripted multi-turn duplex sessions from spoken questions, and record them against a native server."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import benchmarks.duplex.sessions as sessions
from benchmarks.duplex.artifacts import (
    add_load_args,
    add_protocol_args,
    add_server_identity_args,
    server_identity,
)
from benchmarks.duplex.client import SessionInputs
from benchmarks.duplex.run_artifacts import accounting, load_run
from benchmarks.duplex.v15_runner import run_samples

DEFAULT_VOICE = "Samantha"
# note (0xtoward): scripted sessions run for minutes (a camera session fills the thinker context in under two), past the FDB recorders' limit.
SESSION_LIMIT_S = 3600.0


def say(args: argparse.Namespace) -> int:
    question_count = sessions.speak_questions(
        args.questions, args.voices, args.output_dir
    )
    print(f"{question_count} questions in {args.output_dir}")
    return 0


def build(args: argparse.Namespace) -> int:
    seconds = sessions.build_sessions(
        args.questions_dir,
        args.output_dir,
        session_count=args.sessions,
        questions_per_session=args.questions_per_session,
        lead_s=tuple(args.lead_s),
        pause_s=tuple(args.pause_s),
        seed=args.seed,
    )
    print(
        f"{len(seconds)} sessions in {args.output_dir}, {min(seconds)} to {max(seconds)} s"
    )
    return 0


def video(args: argparse.Namespace) -> int:
    seconds = sessions.build_video_sessions(
        args.rows,
        args.videos_root,
        args.output_dir,
        voice=args.voice,
        frame_every_units=args.frame_every_units,
        max_video_s=args.max_video_s,
        answer_pause_s=args.answer_pause_s,
    )
    print(
        f"{len(seconds)} video sessions in {args.output_dir}, {min(seconds)} to {max(seconds)} s"
    )
    return 0


def record(args: argparse.Namespace) -> int:
    frames = (
        ()
        if args.frames_dir is None
        else tuple(path.read_bytes() for path in sorted(args.frames_dir.glob("*.jpg")))
    )
    run_manifest = asyncio.run(
        run_samples(
            args.dataset_root,
            url=args.url,
            output=args.output,
            server=server_identity(
                args.url,
                revision=args.server_revision,
                model=args.model,
                model_revision=args.model_revision,
                runtime=args.runtime,
            ),
            dataset_revision=args.dataset_revision,
            timeout_s=args.timeout,
            profile=args.profile,
            sample_ids=args.sample_id,
            max_per_subset=args.max_sessions,
            dataset=sessions,
            variants=sessions.VARIANTS,
            kind=sessions.RUN_KIND,
            concurrency=args.concurrency,
            turn_detection=args.turn_detection,
            legacy_tail_s=args.legacy_tail,
            session_inputs=SessionInputs(instructions=args.instructions, frames=frames),
            arrival_mean_gap_s=args.arrival_mean_gap_s,
            arrival_seed=args.arrival_seed,
            warmup_wav=args.warmup_wav,
            max_timeout_s=SESSION_LIMIT_S,
        )
    )
    manifest, _, _ = load_run(args.output)
    summary = accounting(manifest, run_manifest)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return (
        0 if summary["variant_status"] == {"pass": summary["selected_variants"]} else 1
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    say_parser = commands.add_parser(
        "say", help="Speak every question line with macOS say"
    )
    say_parser.add_argument("--questions", type=Path, default=sessions.QUESTIONS_PATH)
    say_parser.add_argument(
        "--voices",
        nargs="+",
        default=list(sessions.DEFAULT_VOICES),
        help="Installed voices, listed by: say -v '?'",
    )
    say_parser.add_argument("--output-dir", type=Path, required=True)
    say_parser.set_defaults(handler=say)

    build_parser = commands.add_parser(
        "build", help="Join spoken questions into multi-turn sessions"
    )
    build_parser.add_argument(
        "--questions-dir",
        type=Path,
        required=True,
        help="16 kHz mono PCM16 WAVs, one question each",
    )
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--sessions", type=int, default=32)
    build_parser.add_argument("--questions-per-session", type=int, default=4)
    build_parser.add_argument("--lead-s", type=float, nargs=2, default=(0.0, 6.0))
    build_parser.add_argument("--pause-s", type=float, nargs=2, default=(10.0, 18.0))
    build_parser.add_argument("--seed", type=int, default=20261013)
    build_parser.set_defaults(handler=build)

    video_parser = commands.add_parser(
        "video",
        help="Build camera sessions from Video-MME rows: the video, then its spoken question",
    )
    video_parser.add_argument(
        "--rows",
        type=Path,
        required=True,
        help="Video-MME JSONL with video_path, question_id, question, options and answer",
    )
    video_parser.add_argument(
        "--videos-root",
        type=Path,
        required=True,
        help="Directory the rows' video_path is relative to",
    )
    video_parser.add_argument("--output-dir", type=Path, required=True)
    video_parser.add_argument("--voice", default=DEFAULT_VOICE)
    video_parser.add_argument("--frame-every-units", type=int, default=1)
    video_parser.add_argument("--max-video-s", type=float, default=300.0)
    video_parser.add_argument("--answer-pause-s", type=float, default=15.0)
    video_parser.set_defaults(handler=video)

    record_parser = commands.add_parser(
        "record", help="Record every session WAV of a directory"
    )
    record_parser.add_argument("--dataset-root", type=Path, required=True)
    record_parser.add_argument("--url", required=True, help="Native /v1/realtime URL")
    record_parser.add_argument(
        "--output", type=Path, required=True, help="New immutable run directory"
    )
    add_server_identity_args(record_parser)
    add_protocol_args(record_parser)
    add_load_args(record_parser)
    record_parser.add_argument(
        "--dataset-revision",
        default="scripted",
        help="Label of the session set; every session's digest is recorded anyway",
    )
    record_parser.add_argument(
        "--timeout", type=float, default=600.0, help="Per-session deadline in seconds"
    )
    record_parser.add_argument("--concurrency", type=int, default=1)
    selection = record_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--sample-id", action="append", help="sessions/<name>; repeatable"
    )
    selection.add_argument("--max-sessions", type=int)
    record_parser.add_argument(
        "--instructions", help="Session instructions sent with session.update"
    )
    record_parser.add_argument(
        "--frames-dir",
        type=Path,
        help="JPEG camera frames sent one per unit in name order, cycled, to sessions without their own frames",
    )
    record_parser.set_defaults(handler=record)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
