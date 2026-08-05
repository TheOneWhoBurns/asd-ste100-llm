#!/usr/bin/env python3
"""Download and verify the pinned MLX checkpoint without symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "model.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(destination: Path, lock: dict) -> None:
    symlinks = [path for path in destination.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"symlinks are not permitted: {symlinks}")

    for expected in lock["files"]:
        path = destination / expected["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        if actual_size != expected["bytes"]:
            raise RuntimeError(
                f"{path.name}: expected {expected['bytes']} bytes, got {actual_size}"
            )
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            raise RuntimeError(
                f"{path.name}: expected sha256 {expected['sha256']}, got {actual_hash}"
            )

    config = destination / "config.json"
    index = destination / "model.safetensors.index.json"
    if not config.is_file() or not index.is_file():
        raise RuntimeError("checkpoint metadata is incomplete")
    print(f"Verified pinned model at {destination}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    destination = ROOT / lock["destination"]
    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=lock["repository"],
            revision=lock["revision"],
            local_dir=destination,
        )
    verify(destination, lock)


if __name__ == "__main__":
    main()

