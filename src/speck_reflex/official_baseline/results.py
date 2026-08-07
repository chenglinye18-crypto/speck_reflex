"""Result schema and checkpoint integrity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED_RESULT_KEYS = {"mode", "seed", "sample", "sinabs_software", "dynapcnn", "samna_config", "specksim", "hardware"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_results(data: dict[str, object]) -> None:
    missing = REQUIRED_RESULT_KEYS - data.keys()
    if missing:
        raise ValueError(f"missing result keys: {sorted(missing)}")
    if data["mode"] not in {"pipeline_smoke", "functional_nmnist"}:
        raise ValueError("unsupported result mode")


def write_results(path: Path, data: dict[str, object]) -> None:
    validate_results(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
