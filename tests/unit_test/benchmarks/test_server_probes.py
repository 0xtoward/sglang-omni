"""Server-side views of a serving run and the command that records them around a benchmark."""

import json
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue

from benchmarks.metrics import server_probes


def write_code2wav_batches(event_dir: Path) -> None:
    """Code2Wav runs a and b in one batch, then c alone."""
    events = [
        ("a", "stage_dispatch", 5.0, {}),
        ("b", "stage_dispatch", 8.0, {}),
        ("a", "scheduler_batch_start", 10.0, {"batch_size": 2, "gather_ms": 2.0}),
        ("b", "scheduler_batch_start", 10.0, {"batch_size": 2, "gather_ms": 2.0}),
        ("a", "scheduler_batch_end", 30.0, {"batch_size": 2}),
        ("b", "scheduler_batch_end", 30.0, {"batch_size": 2}),
        ("c", "stage_dispatch", 40.0, {}),
        ("c", "scheduler_batch_start", 40.0, {"batch_size": 1, "gather_ms": 0.0}),
        ("c", "scheduler_batch_end", 50.0, {"batch_size": 1}),
    ]
    (event_dir / "events_code2wav_23.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "request_id": request_id,
                    "stage": "code2wav",
                    "event_name": event_name,
                    "timestamp_ns": int(time_ms * 1e6),
                    "metadata": metadata,
                }
            )
            + "\n"
            for request_id, event_name, time_ms, metadata in events
        )
    )


def test_simple_stage_batches_count_sizes_waits_and_busy_share(tmp_path: Path) -> None:
    write_code2wav_batches(tmp_path)

    batches = server_probes.simple_stage_batches(tmp_path)["code2wav"]

    assert (batches["batches"], batches["requests_per_batch_max"]) == (2, 2)
    assert batches["requests_per_batch_mean"] == pytest.approx(1.5)
    assert batches["requests_alone_percent"] == pytest.approx(100 / 3), "c ran alone"
    assert batches["gather_ms"]["p50"] == pytest.approx(1.0)
    assert batches["wait_ms"] == pytest.approx(
        {"count": 3, "p50": 2.0, "p90": 4.4}
    ), "a, b and c waited 5, 2 and 0 ms"
    assert batches["compute_ms"] == pytest.approx(
        {"count": 2, "p50": 15.0, "p90": 19.0}
    ), "the batches computed for 20 and 10 ms"
    assert batches["busy_percent"] == pytest.approx(75.0), "30 of the 40 ms computed"


class RecordingProfileServer:
    """Takes the request profile commands, noting whether the command already ran; at stop it writes the Code2Wav batches and then its run status, as each server process does after the stop."""

    def __init__(self, command_marker: Path) -> None:
        self.command_marker = command_marker
        self.calls: list[tuple[str, bool]] = []
        self.event_dir: Path | None = None

    def __call__(self, url: str, body: dict[str, JsonValue]) -> None:
        self.calls.append((url, self.command_marker.exists()))
        if url.endswith("/start_request_profile"):
            self.event_dir = Path(body["event_dir"])
        else:
            self.event_dir.mkdir()
            write_code2wav_batches(self.event_dir)
            (self.event_dir / "run_status_23.json").write_text("{}")


def test_probe_command_runs_between_profile_start_and_stop_then_prints_the_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command_marker = tmp_path / "command-ran"
    server = RecordingProfileServer(command_marker)
    monkeypatch.setattr(server_probes, "post_profile_command", server)
    monkeypatch.setattr(server_probes, "EVENT_FILES_SETTLE_S", 0.0)
    output_dir = tmp_path / "tts-c4"
    command = [
        sys.executable,
        "-c",
        "import pathlib, sys; pathlib.Path(sys.argv[1]).touch(); sys.exit(3)",
    ]

    code = server_probes.main(
        ["--server-url", "http://server", "--output-dir", str(output_dir), "--"]
        + [*command, str(command_marker)]
    )

    assert code == 3, "the command's exit status"
    assert server.calls == [
        ("http://server/start_request_profile", False),
        ("http://server/stop_request_profile", True),
    ]
    views = json.loads((output_dir / "server.json").read_text())
    assert views["simple_stage_batches"]["code2wav"]["batches"] == 2
    assert views["stages"] == {}, "no request reached the coordinator"
    assert "| code2wav | 1.5 (2) / 33% / 75% / 4 ms |" in capsys.readouterr().out
