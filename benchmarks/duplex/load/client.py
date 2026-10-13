# SPDX-License-Identifier: Apache-2.0
"""Open-loop full-duplex load: sessions arrive as a Poisson process and stream WAV input at real time.

Each session records the send time of every 80 ms input packet and the arrival time of every server event,
so report.py can rebuild each unit's lag offline.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import statistics
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import websockets
from pydantic import JsonValue

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
PCM16_SAMPLE_BYTES = 2
PACKET_SAMPLES = 1280
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
CONNECT_TIMEOUT_S = 60.0
INLINE_AUDIO_CHARS = 200
DEFAULT_INSTRUCTIONS = "You are a helpful voice assistant. Answer briefly."

Event = dict[str, JsonValue]


@dataclass(kw_only=True, frozen=True)
class SessionSettings:
    port: int
    instructions: str
    silence_tail_s: float
    output_dir: Path
    save_audio: bool


def read_pcm16(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as source:
        layout = (source.getnchannels(), source.getframerate(), source.getsampwidth())
        if layout != (1, INPUT_SAMPLE_RATE, PCM16_SAMPLE_BYTES):
            raise ValueError(f"{wav_path} is {layout}, not 16 kHz mono PCM16")
        else:
            return source.readframes(source.getnframes())


def without_audio_payload(event: Event) -> Event:
    """Keep an event's fields, with inline base64 audio replaced by its byte count."""
    kept: Event = {}
    for field_name, value in event.items():
        if (
            field_name in ("audio", "delta")
            and isinstance(value, str)
            and len(value) > INLINE_AUDIO_CHARS
        ):
            kept[f"{field_name}_bytes"] = len(base64.b64decode(value))
        elif field_name == "session" and isinstance(value, dict):
            kept[field_name] = {
                name: setting
                for name, setting in value.items()
                if name != "reference_audio"
            }
        else:
            kept[field_name] = value
    return kept


def arrival_offsets_s(
    session_count: int,
    *,
    arrival: Literal["poisson", "fixed"],
    mean_gap_s: float,
    seed: int,
) -> list[float]:
    """Start time of every session: a Poisson process with this mean gap, or evenly spaced."""
    if arrival == "poisson":
        generator = random.Random(seed)
        offsets_s, clock_s = [], 0.0
        for _ in range(session_count):
            offsets_s.append(clock_s)
            clock_s += generator.expovariate(1.0 / mean_gap_s)
        return offsets_s
    else:
        return [index * mean_gap_s for index in range(session_count)]


async def run_session(index: int, wav_path: Path, settings: SessionSettings) -> Event:
    silence = bytes(
        int(INPUT_SAMPLE_RATE * settings.silence_tail_s) * PCM16_SAMPLE_BYTES
    )
    pcm = read_pcm16(wav_path) + silence
    sends: list[Event] = []
    events: list[Event] = []
    reply_audio = bytearray()
    transcript: list[str] = []
    error: str | None = None
    start_time = time.perf_counter()

    async def stream_input(websocket: websockets.ClientConnection) -> None:
        packet_bytes = PACKET_SAMPLES * PCM16_SAMPLE_BYTES
        packet_period_s = PACKET_SAMPLES / INPUT_SAMPLE_RATE
        next_send_time = time.perf_counter()
        for sequence, offset in enumerate(range(0, len(pcm), packet_bytes)):
            packet = base64.b64encode(pcm[offset : offset + packet_bytes]).decode()
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": f"audio-{sequence}",
                        "audio": packet,
                        "sglang": {"seq": sequence},
                    }
                )
            )
            sent_s = (offset + packet_bytes) / PCM16_SAMPLE_BYTES / INPUT_SAMPLE_RATE
            sends.append(
                {
                    "seq": sequence,
                    "t": round(time.perf_counter() - start_time, 4),
                    "input_s": round(sent_s, 3),
                }
            )
            next_send_time += packet_period_s
            await asyncio.sleep(max(0.0, next_send_time - time.perf_counter()))
        await websocket.send(
            json.dumps({"type": "sglang.input_audio.end", "event_id": "end"})
        )

    streamer: asyncio.Task[None] | None = None
    try:
        async with websockets.connect(
            f"ws://127.0.0.1:{settings.port}/v1/realtime",
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=CONNECT_TIMEOUT_S,
        ) as websocket:
            async for message in websocket:
                event: Event = json.loads(message)
                record: Event = {
                    "t": round(time.perf_counter() - start_time, 4),
                    **without_audio_payload(event),
                }
                event_type = event.get("type")
                delta = event.get("delta")
                if event_type == "response.output_audio.delta" and isinstance(
                    delta, str
                ):
                    chunk = base64.b64decode(delta)
                    reply_audio += chunk
                    record["audio_s"] = round(
                        len(chunk) / PCM16_SAMPLE_BYTES / OUTPUT_SAMPLE_RATE, 4
                    )
                elif (
                    event_type == "response.output_audio_transcript.delta"
                    and isinstance(delta, str)
                ):
                    transcript.append(delta)
                else:
                    pass
                events.append(record)
                if event_type == "session.created":
                    session_update = {
                        "instructions": settings.instructions,
                        "output_modalities": ["audio"],
                    }
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "session.update",
                                "event_id": "config",
                                "session": session_update,
                            }
                        )
                    )
                elif event_type == "session.updated":
                    streamer = asyncio.create_task(stream_input(websocket))
                elif event_type == "sglang.input_audio.drained":
                    await websocket.send(
                        json.dumps({"type": "session.close", "event_id": "close"})
                    )
                elif event_type == "session.closed":
                    break
                elif event_type == "error":
                    error = json.dumps(event.get("error"))
                    break
                else:
                    pass
    except (OSError, websockets.WebSocketException) as exc:
        # note (0xtoward): a refused or dropped connection is a measured failure of this session, not of the load.
        error = repr(exc)
    finally:
        if streamer is not None and not streamer.done():
            streamer.cancel()
        else:
            pass

    audio_times_s = [
        record["t"]
        for record in events
        if record.get("type") == "response.output_audio.delta"
    ]
    audio_gaps_s = [
        round(current - previous, 4)
        for previous, current in zip(audio_times_s, audio_times_s[1:])
    ]
    tag = f"s{index:02d}"
    if reply_audio and settings.save_audio:
        with wave.open(str(settings.output_dir / f"{tag}.wav"), "wb") as output:
            output.setparams(
                (1, PCM16_SAMPLE_BYTES, OUTPUT_SAMPLE_RATE, 0, "NONE", "not compressed")
            )
            output.writeframes(bytes(reply_audio))
    else:
        pass
    event_types = sorted({str(record.get("type")) for record in events})
    summary: Event = {
        "session": index,
        "wav": wav_path.name,
        "error": error,
        "input_s": round(len(pcm) / PCM16_SAMPLE_BYTES / INPUT_SAMPLE_RATE, 3),
        "total_wall_s": round(time.perf_counter() - start_time, 3),
        "output_audio_s": round(
            len(reply_audio) / PCM16_SAMPLE_BYTES / OUTPUT_SAMPLE_RATE, 3
        ),
        "audio_deltas": len(audio_times_s),
        "first_audio_t": audio_times_s[0] if audio_times_s else None,
        "last_audio_t": audio_times_s[-1] if audio_times_s else None,
        "audio_gap_max_s": max(audio_gaps_s) if audio_gaps_s else None,
        "event_counts": {
            event_type: sum(1 for record in events if record.get("type") == event_type)
            for event_type in event_types
        },
        "transcript": "".join(transcript),
    }
    with open(settings.output_dir / f"{tag}.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "sends": sends, "events": events},
            handle,
            indent=1,
            ensure_ascii=False,
        )
    return summary


async def run_load(
    wav_paths: list[Path], offsets_s: list[float], settings: SessionSettings
) -> list[Event]:
    async def start_later(index: int) -> Event:
        await asyncio.sleep(offsets_s[index])
        return await run_session(index, wav_paths[index % len(wav_paths)], settings)

    return await asyncio.gather(
        *(start_later(index) for index in range(len(offsets_s)))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--wav",
        type=Path,
        nargs="+",
        required=True,
        help="16 kHz mono PCM16 inputs; session i streams the i-th, cycling",
    )
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--arrival", choices=("poisson", "fixed"), default="poisson")
    parser.add_argument(
        "--mean-gap-s",
        type=float,
        default=3.0,
        help="mean gap between session starts (the fixed gap with --arrival fixed)",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--silence-tail-s",
        type=float,
        default=4.0,
        help="silence appended to each input so the last answer can finish",
    )
    parser.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = SessionSettings(
        port=args.port,
        instructions=args.instructions,
        silence_tail_s=args.silence_tail_s,
        output_dir=args.output_dir,
        save_audio=args.save_audio,
    )
    offsets_s = arrival_offsets_s(
        args.sessions, arrival=args.arrival, mean_gap_s=args.mean_gap_s, seed=args.seed
    )
    begin = time.perf_counter()
    summaries = asyncio.run(run_load(args.wav, offsets_s, settings))
    completed = [summary for summary in summaries if summary["error"] is None]
    first_audio_times_s = [
        summary["first_audio_t"]
        for summary in completed
        if summary["first_audio_t"] is not None
    ]
    totals: Event = {
        "sessions": len(summaries),
        "failed": len(summaries) - len(completed),
        "wall_s": round(time.perf_counter() - begin, 3),
        "input_s_total": round(sum(summary["input_s"] for summary in completed), 3),
        "output_audio_s_total": round(
            sum(summary["output_audio_s"] for summary in completed), 3
        ),
        "first_audio_t_mean": (
            round(statistics.fmean(first_audio_times_s), 3)
            if first_audio_times_s
            else None
        ),
        "start_offsets_s": [round(offset_s, 2) for offset_s in offsets_s],
        "errors": [summary["error"] for summary in summaries if summary["error"]],
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"totals": totals, "sessions": summaries},
            handle,
            indent=1,
            ensure_ascii=False,
        )
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
