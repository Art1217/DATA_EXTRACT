"""Corre el pipeline con un modelo real y guarda sus respuestas para evaluarlas.

Usa el proveedor configurado en `.env` (`AI_PROVIDER=gemini_api` o `vertex`) y
graba cada respuesta del modelo en `evals/<documento>/<nombre>.json`, con la
misma forma que lee `run_eval.py`. Después se mide sin volver a pagar:

    python evals/record.py goldfields_2024 gemini data/real/goldfields_2024.pdf
    python evals/run_eval.py goldfields_2024 gemini

Cada llamada queda además, con su texto crudo, en
`evals/<documento>/out/<nombre>/calls.jsonl`, para revisar qué respondió el
modelo cuando algo no cuadra.

Por defecto (`--mode auto`) lanza `POST /auto`: plan, identidad y el pipeline
completo de cada análisis detectado, en un solo job y por eso en una sola traza
de LangSmith. Lo que el plan no cubre (el dictamen) se busca aparte.
`--mode stages` llama etapa por etapa y solo extrae lo indicado en `--extract`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

_PAGE_LABEL = re.compile(r"IMAGEN PÁGINA (\d+) START")
DEFAULT_EXTRACT = ("situacion_financiera", "resultados", "flujo_efectivo")


def _parse(text: str | None):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class Recorder:
    """Envuelve al proveedor real y acumula sus respuestas por etapa y prompt."""

    def __init__(self, inner, model_label: str, log_path: Path):
        self.inner = inner
        self.lock = threading.Lock()
        self.log_path = log_path
        self.answers: dict = {
            "model": model_label,
            "classification": {},
            "identity": {},
            "search": {},
            "extraction": {},
            "quality": {},
            "unparsed": [],
        }

    def generate(self, request):
        started = time.time()
        response = self.inner.generate(request)
        pages = [
            int(m.group(1))
            for part in request.parts
            if part.text
            for m in [_PAGE_LABEL.search(part.text)]
            if m
        ]
        parsed = _parse(response.text)
        with self.lock:
            self._store(request.task, request.prompt_name, parsed, response.text)
            with self.log_path.open("a", encoding="utf-8") as log:
                entry = {
                    "task": request.task,
                    "prompt": request.prompt_name,
                    "model": request.model,
                    "pages": [pages[0], pages[-1]] if pages else None,
                    "seconds": round(time.time() - started, 1),
                    "finish_reason": response.finish_reason,
                    "usage": asdict(response.usage) if response.usage else None,
                    "text": response.text,
                }
                log.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return response

    def _store(self, task, name, parsed, raw):
        if parsed is None:
            self.answers["unparsed"].append({"task": task, "prompt": name, "text": raw})
            return
        a = self.answers
        if task == "classify":
            items = parsed.get("clasificaciones", []) if isinstance(parsed, dict) else parsed
            a["classification"].setdefault(name, {"clasificaciones": []})["clasificaciones"].extend(
                items or []
            )
        elif task == "search":
            slot = a["search"].setdefault(name, {"relevant_sections": [], "alerts": []})
            slot["relevant_sections"].extend(parsed.get("relevant_sections", []))
            slot["alerts"].extend(parsed.get("alerts", []))
        elif task == "identity":
            a["identity"] = parsed
        elif task == "extract":
            rows = parsed.get("filas", []) if isinstance(parsed, dict) else parsed
            a["extraction"].setdefault(name, {"filas": []})["filas"].extend(rows or [])
        elif task == "quality":
            a["quality"][name] = parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("document", help="Carpeta dentro de evals/, por ejemplo goldfields_2024.")
    parser.add_argument("name", help="Nombre del archivo de respuestas, por ejemplo gemini.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--mode",
        choices=("auto", "stages"),
        default="auto",
        help="auto: un solo job (una traza por documento). stages: etapa por etapa.",
    )
    parser.add_argument(
        "--extract",
        nargs="*",
        default=list(DEFAULT_EXTRACT),
        help="Solo con --mode stages: qué análisis extraer.",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # Storage y temporales aislados: la corrida no ensucia ./Storage ni
    # reutiliza resultados viejos del mismo PDF.
    os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="record_storage_")
    os.environ["TEMP_DIR"] = tempfile.mkdtemp(prefix="record_temp_")

    from fastapi.testclient import TestClient

    from app.adapters import llm
    from app.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()
    if settings.ai_provider.value == "fake":
        sys.exit("AI_PROVIDER es 'fake': configura gemini_api o vertex en .env antes de grabar.")

    out_dir = ROOT / args.document / "out" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "calls.jsonl"
    log_path.write_text("", encoding="utf-8")

    real = llm.get_provider()
    label = f"{settings.ai_provider.value}: {settings.model_name_classify} / {settings.model_name_extract}"
    recorder = Recorder(real, label, log_path)
    llm.get_provider = lambda: recorder

    from app.domain.analysis import ANALYSIS_TYPES
    from app.main import app

    def wait(client, job, what):
        print(f"  {what}...", flush=True)
        while True:
            state = client.get(f"/v1/jobs/{job['job_id']}").json()
            if state["status"] in {"succeeded", "failed", "cancelled"}:
                if state["status"] != "succeeded":
                    print(f"    {state['status']}: {state.get('error')}")
                return state
            time.sleep(1)

    print(f"Grabando {args.pdf.name} con {label}")
    with TestClient(app) as client, args.pdf.open("rb") as handle:
        doc_id = client.post(
            "/v1/documents", files={"file": (args.pdf.name, handle, "application/pdf")}
        ).json()["document_id"]
        base = f"/v1/documents/{doc_id}"

        if args.mode == "auto":
            # Un solo job: plan, identidad y el pipeline de cada análisis que
            # el plan detecte, todo bajo una misma traza en LangSmith.
            state = wait(client, client.post(f"{base}/auto", json={}).json(), "flujo automático")
            planned = set((state.get("result") or {}).get("plan", {}).get("analysis_types", []))
            pending = [key for key in ANALYSIS_TYPES if key not in planned]
        else:
            wait(client, client.post(f"{base}/plan", json={}).json(), "clasificación (Fase 1 y 2)")
            wait(client, client.post(f"{base}/identity", json={}).json(), "identidad")
            for key in args.extract:
                body = {"analysis_type": key}
                wait(client, client.post(f"{base}/pipeline", json=body).json(), f"pipeline {key}")
            pending = [key for key in ANALYSIS_TYPES if key not in args.extract]

        # Lo que no pasó por un pipeline (el dictamen, que solo se localiza, o
        # lo que el plan no detectó) se busca igual: la búsqueda también se mide.
        for key in pending:
            body = {"analysis_type": key, "cut": False}
            wait(client, client.post(f"{base}/search", json=body).json(), f"búsqueda {key}")

    target = ROOT / args.document / f"{args.name}.json"
    target.write_text(
        json.dumps(recorder.answers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    unparsed = len(recorder.answers["unparsed"])
    print(f"\nRespuestas en {target}" + (f" ({unparsed} sin JSON válido)" if unparsed else ""))
    print(f"Llamadas crudas en {log_path}")

    totals: dict[str, dict[str, int]] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        model = totals.setdefault(entry["model"], {"llamadas": 0})
        model["llamadas"] += 1
        for field, count in (entry.get("usage") or {}).items():
            model[field] = model.get(field, 0) + count
    for model, counts in totals.items():
        print(f"Tokens {model}: {counts}")
    print(f"Medir: python evals/run_eval.py {args.document} {args.name}")


if __name__ == "__main__":
    main()
