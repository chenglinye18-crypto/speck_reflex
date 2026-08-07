#!/usr/bin/env python3
"""Verify repository hygiene without hardware or network access."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speck_reflex.official_baseline.results import validate_results  # noqa: E402


def tracked() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True)
    return [ROOT / line for line in out.splitlines() if line and not line.startswith("third_party/")]


def main() -> int:
    errors: list[str] = []
    files = tracked()
    forbidden_parts = {".venv", "data/official", "data/raw"}
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if any(relative == part or relative.startswith(part + "/") for part in forbidden_parts):
            errors.append(f"tracked generated/private path: {relative}")
        if path.suffix == ".bin":
            errors.append(f"Flash binary must not be tracked: {relative}")
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".nir"}:
            errors.append(f"model file requires explicit provenance review before tracking: {relative}")
        if path.exists() and path.is_file() and path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"large file (>10 MiB): {relative}")
        if path.suffix in {".py", ".md", ".sh", ".toml", ".yaml", ".yml", ".txt", ".json"} and path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if "/home/" + "speck" in text:
                errors.append(f"hard-coded local path: {relative}")
            if path.suffix in {".py", ".sh"} and re.search(r"io_sel\s*=\s*24", text):
                errors.append(f"unexplained io_sel default: {relative}")
            forbidden_write = "Flash" + "Write"
            if path.suffix in {".py", ".sh"} and forbidden_write in text:
                errors.append(f"forbidden Flash write API token: {relative}")
            forbidden_open = "open" + "_device"
            if relative.startswith("tests/") and forbidden_open in text:
                errors.append(f"device open in test: {relative}")

    for result_path in (ROOT / "experiments/official_baselines/nmnist/artifacts").glob("*results.json"):
        try:
            validate_results(json.loads(result_path.read_text(encoding="utf-8")))
        except Exception as exc:
            errors.append(f"invalid results {result_path.relative_to(ROOT)}: {exc}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\((?!https?://|#)([^)]+)\)", readme):
        clean = target.split("#", 1)[0]
        if clean and not (ROOT / clean).exists():
            errors.append(f"broken README link: {target}")

    submodules = subprocess.run(["git", "submodule", "status", "--recursive"], cwd=ROOT, text=True, capture_output=True, check=False)
    if submodules.returncode or any(line.startswith(("-", "+", "U")) for line in submodules.stdout.splitlines()):
        errors.append("submodules are missing, modified, or unresolved")

    if errors:
        print("REPOSITORY_VERIFICATION_FAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"checked_files: {len(files)}")
    print("hardware_accessed: False")
    print("REPOSITORY_VERIFICATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
