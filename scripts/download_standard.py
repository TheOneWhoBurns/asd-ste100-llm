#!/usr/bin/env python3
"""Download the official ASD-STE100 Issue 9 PDF into ignored local storage."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "standard.lock.json"


def main() -> None:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    destination = ROOT / lock["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        lock["source"],
        headers={"User-Agent": "asd-ste100-local-research/0.1"},
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()

    if not payload.startswith(b"%PDF-"):
        raise RuntimeError("official source did not return a PDF")

    temporary = destination.with_suffix(".pdf.part")
    temporary.write_bytes(payload)
    temporary.replace(destination)

    digest = hashlib.sha256(payload).hexdigest()
    if len(payload) != lock["bytes"] or digest != lock["sha256"]:
        raise RuntimeError(
            "the official PDF changed; inspect it before updating standard.lock.json"
        )

    reader = PdfReader(destination)
    if len(reader.pages) < 100:
        raise RuntimeError(f"unexpected page count: {len(reader.pages)}")

    print(
        f"Downloaded Issue {lock['issue']} to {destination} "
        f"({len(reader.pages)} pages, {len(payload)} bytes, sha256={digest})"
    )


if __name__ == "__main__":
    main()
