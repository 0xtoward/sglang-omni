# SPDX-License-Identifier: Apache-2.0
"""Research capture of Drama 3 rows for draft-model studies, on when DRAMA3_SPEC_CAPTURE_DIR is set.

Every forward appends, per row, the request id, position, input frame and final-norm hidden state; rows that sample a frame
also keep the top-64 raw semantic logits, the top-64 code 1 logits and the sampled frame. Needs eager decode (no CUDA graph).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import torch

CAPTURE_DIR_ENV = "DRAMA3_SPEC_CAPTURE_DIR"
TOP_LOGITS = 64
FLUSH_PERIOD_S = 10.0


class SpecCapture:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, object]] = []
        self.part = 0
        self.code_one_logits: torch.Tensor | None = None
        self.lock = threading.Lock()
        # note (0xtoward): the server is stopped with a signal, so a timer writes what an idle server holds.
        threading.Thread(target=self.flush_periodically, daemon=True).start()

    def keep_code_one_logits(
        self, module: torch.nn.Module, inputs: tuple, output: torch.Tensor
    ) -> None:
        self.code_one_logits = output.detach()

    def record(
        self,
        kind: str,
        request_ids: list[str],
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        hidden_states: torch.Tensor,
        sampling_rows: torch.Tensor | None,
        semantic_logits: torch.Tensor | None,
        sampled_frames: torch.Tensor | None,
    ) -> None:
        entry: dict[str, object] = {
            "kind": kind,
            "request_ids": request_ids,
            "positions": positions.to("cpu", torch.int32),
            "input_frames": input_frames.to("cpu", torch.int32),
            "hidden_states": hidden_states.to("cpu", torch.bfloat16),
        }
        if sampling_rows is not None and semantic_logits is not None:
            main_values, main_ids = semantic_logits.float().topk(TOP_LOGITS, dim=-1)
            code_one = self.code_one_logits
            entry["sampling_rows"] = sampling_rows.to("cpu", torch.int32)
            entry["main_values"] = main_values.to("cpu", torch.float16)
            entry["main_ids"] = main_ids.to("cpu", torch.int16)
            entry["sampled_frames"] = sampled_frames.to("cpu", torch.int32)
            if code_one is not None and code_one.shape[0] == semantic_logits.shape[0]:
                code_values, code_ids = code_one.float().topk(TOP_LOGITS, dim=-1)
                entry["code_one_values"] = code_values.to("cpu", torch.float16)
                entry["code_one_ids"] = code_ids.to("cpu", torch.int16)
            else:
                pass
        else:
            pass
        self.code_one_logits = None
        with self.lock:
            self.entries.append(entry)

    def flush_periodically(self) -> None:
        while True:
            time.sleep(FLUSH_PERIOD_S)
            with self.lock:
                entries, self.entries = self.entries, []
            if entries:
                torch.save(entries, self.directory / f"part-{os.getpid()}-{self.part:05d}.pt")
                self.part += 1
            else:
                pass


def capture_from_environment() -> SpecCapture | None:
    directory = os.environ.get(CAPTURE_DIR_ENV)
    if directory:
        return SpecCapture(Path(directory))
    else:
        return None
