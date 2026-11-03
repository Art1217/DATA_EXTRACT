"""Descarga los PDFs públicos que usan las evaluaciones.

Los PDFs no se versionan: son documentos de terceros, publicados por cada
empresa y por la SMV. Cada `evals/<documento>/expected.json` guarda de dónde
salió el suyo, y este script los baja a `data/real/`:

    python evals/download.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parent / "data" / "real"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for expected in sorted(ROOT.glob("*/expected.json")):
        meta = json.loads(expected.read_text(encoding="utf-8"))
        destination = TARGET / meta["document"]
        if destination.exists():
            print(f"ya existe   {destination.name}")
            continue
        response = httpx.get(
            meta["source"],
            follow_redirects=True,
            timeout=120,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            sys.exit(f"{meta['source']} no devolvió un PDF; descárgalo a mano en {destination}.")
        destination.write_bytes(response.content)
        print(f"descargado  {destination.name} ({len(response.content) // 1024} KB)")


if __name__ == "__main__":
    main()
