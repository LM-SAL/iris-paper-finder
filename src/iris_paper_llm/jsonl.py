"""Shared JSON Lines file helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict]:
    """Read records, skipping a last line left unparsable by an interrupted append."""
    lines = [
        (number, line) for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()
    ]
    records = []
    for number, line in lines:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == lines[-1][0]:
                logger.warning("Skipping unparsable last line %d of %s", number, path)
                break
            msg = f"{path}:{number}: invalid JSON: {error}"
            raise ValueError(msg) from error
    return records


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    """Append one record, first terminating or dropping an unterminated last line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        end = stream.seek(0, os.SEEK_END)
        stream.seek(max(end - 1, 0))
        if end and stream.read(1) != b"\n":
            stream.seek(0)
            content = stream.read()
            start = content.rfind(b"\n") + 1
            try:
                json.loads(content[start:])
            except ValueError:
                # A fragment from an interrupted append is never a usable record.
                logger.warning("Dropping an unterminated partial last line from %s", path)
                stream.truncate(start)
            else:
                stream.write(b"\n")
        stream.write(json.dumps(record, separators=(",", ":")).encode() + b"\n")


def atomic_write(path: Path, content: bytes) -> None:
    """Replace a file so that readers and interrupted runs never see it partly written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Atomically replace a JSONL file."""
    atomic_write(path, "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode())
