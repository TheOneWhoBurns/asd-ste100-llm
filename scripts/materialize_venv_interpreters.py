#!/usr/bin/env python3
"""Replace virtual-environment interpreter links with ordinary file copies."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
BIN_DIR = PROJECT_ROOT / ".venv" / "bin"
INTERPRETERS = ("python", "python3", "python3.12")


def _venv_configuration() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (VENV_DIR / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _copy_runtime_library() -> None:
    configuration = _venv_configuration()
    version = configuration["version_info"].split(".")
    library_name = f"libpython{version[0]}.{version[1]}.dylib"
    source = Path(configuration["home"]).parent / "lib" / library_name
    destination = VENV_DIR / "lib" / library_name
    if source.is_file() and not destination.is_file():
        shutil.copy2(source, destination)


def main() -> None:
    _copy_runtime_library()
    for name in INTERPRETERS:
        destination = BIN_DIR / name
        if not destination.is_symlink():
            continue
        source = destination.resolve(strict=True)
        temporary = BIN_DIR / f".{name}.materialized"
        shutil.copy2(source, temporary, follow_symlinks=True)
        os.replace(temporary, destination)


if __name__ == "__main__":
    main()
