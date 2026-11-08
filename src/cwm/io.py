"""Small helpers for JSON Lines files and run metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write atomically, so an interrupted run never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def environment_info() -> dict[str, Any]:
    """Versions and hardware, saved with every run for reproducibility."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        import transformers

        info["torch"] = torch.__version__
        info["transformers"] = transformers.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        info["git_commit"] = commit
    except (OSError, subprocess.CalledProcessError):
        info["git_commit"] = None
    return info
