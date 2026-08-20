from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any
import csv
import json

import torch


def prepare_run_dir(base: str | Path, run_name: str) -> Path:
    root = Path(base)
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_round_model(run_dir: str | Path, round_index: int, state: "OrderedDict[str, torch.Tensor]") -> Path:
    models_dir = Path(run_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"round_{round_index:04d}.pt"
    torch.save(state, path)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_rounds_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty round table")
    columns = list(rows[0].keys())
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

