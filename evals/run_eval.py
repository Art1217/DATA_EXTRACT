"""Evalúa las respuestas de un modelo contra los valores verificados de un documento.

Cada carpeta de `evals/` es un documento real: `expected.json` tiene la
respuesta correcta, verificada a mano contra el PDF, y cada `<modelo>.json`
guarda lo que respondió un modelo a cada prompt.

El script hace dos cosas:

1. **Mide** cada etapa contra `expected.json`: precisión y recall de páginas
   en clasificación y búsqueda, campos de la identidad y cifras extraídas.
2. **Reproduce** el pipeline real con esas respuestas (`--pipeline`): un
   proveedor de reemplazo devuelve lo guardado en vez de llamar a un modelo, así
   que el recorte, las reglas contables y el Excel corren sobre el PDF de verdad.

Así se comparan modelos sin pagar dos veces: la corrida de Gemini se guarda una
vez y se puede volver a medir cada vez que cambie el código.

Uso:
    python evals/run_eval.py goldfields_2024 simulacion
    python evals/run_eval.py goldfields_2024 simulacion --pipeline data/real/goldfields_2024.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from app.adapters.llm import LLMResponse, Request  # noqa: E402

_PAGE_LABEL = re.compile(r"IMAGEN PÁGINA (\d+) START")


# --------------------------------------------------------------------------
# Medición
# --------------------------------------------------------------------------


def _pages_by_label(payload: dict, field: str) -> dict[str, set[int]]:
    found: dict[str, set[int]] = {}
    for item in payload.get("clasificaciones", []):
        found.setdefault(str(item.get(field)), set()).add(int(item["page_number"]))
    return found


def _prf(predicted: set[int], required: set[int], acceptable: set[int] = frozenset()) -> dict:
    """Recall sobre las páginas obligatorias; precisión tolerando las aceptables."""
    hits = predicted & required
    wrong = predicted - required - acceptable
    recall = len(hits) / len(required) if required else 1.0
    precision = (len(predicted) - len(wrong)) / len(predicted) if predicted else 0.0
    return {
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "faltan": sorted(required - predicted),
        "sobran": sorted(wrong),
    }


def _norm(text) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _cells(rows: list[dict]) -> dict[tuple, float]:
    """`{(seccion, concepto, periodo): valor}`; la sección distingue filas homónimas."""
    cells = {}
    for row in rows:
        key = (_norm(row.get("seccion")), _norm(row.get("concepto")), str(row.get("periodo")))
        cells.setdefault(key, row.get("valor"))
    return cells


def score(expected: dict, answers: dict) -> dict:
    report: dict = {"classification": {}, "search": {}, "identity": {}, "extraction": {}}

    fields = {"grupo": "grupo", "estados": "codigo", "notas": "codigo"}
    for prompt, field in fields.items():
        predicted = _pages_by_label(answers.get("classification", {}).get(prompt, {}), field)
        for label, pages in expected["classification"][prompt].items():
            report["classification"][f"{prompt}:{label}"] = _prf(
                predicted.get(label, set()), set(pages)
            )

    for key, truth in expected["search"].items():
        payload = answers.get("search", {}).get(key)
        if payload is None:
            continue
        predicted = {int(s["page_number"]) for s in payload.get("relevant_sections", [])}
        report["search"][key] = _prf(
            predicted, set(truth["required"]), set(truth.get("acceptable", []))
        )

    got = answers.get("identity", {})
    for field, value in expected["identity"].items():
        ok = _norm(got.get(field)) == _norm(value)
        report["identity"][field] = (
            "ok" if ok else f"esperado {value!r}, obtenido {got.get(field)!r}"
        )

    for key, truth_rows in expected.get("extraction", {}).items():
        payload = answers.get("extraction", {}).get(key)
        if payload is None:
            continue
        truth, got_cells = _cells(truth_rows), _cells(payload.get("filas", []))
        wrong = {
            k: (v, got_cells[k]) for k, v in truth.items() if k in got_cells and got_cells[k] != v
        }
        missing = [k for k in truth if k not in got_cells]
        extra = [k for k in got_cells if k not in truth]

        # Una celda que no casó por nombre pero tiene el mismo período y la
        # misma cifra que una sobrante es la misma fila con otra etiqueta: la
        # sección o un subtotal sin título no están impresos, cada uno los
        # nombra a su manera. La cifra es lo que se mide; la etiqueta se avisa.
        relabeled = []
        for k in list(missing):
            match = next(
                (e for e in extra if e[2] == k[2] and got_cells[e] == truth[k]),
                None,
            )
            if match is not None:
                missing.remove(k)
                extra.remove(match)
                relabeled.append(f"{k[0]} / {k[1]} -> {match[0]} / {match[1]} ({k[2]})")

        correct = len(truth) - len(wrong) - len(missing)
        report["extraction"][key] = {
            "celdas": len(truth),
            "exactitud": round(correct / len(truth), 3),
            "cifras_distintas": [
                f"{k[1]} {k[2]}: esperado {a}, obtenido {b}" for k, (a, b) in wrong.items()
            ],
            "faltan": [f"{k[1]} {k[2]}" for k in missing],
            "sobran": [f"{k[1]} {k[2]}" for k in extra],
            "etiqueta_distinta": relabeled,
        }
    return report


# --------------------------------------------------------------------------
# Reproducción del pipeline
# --------------------------------------------------------------------------


class ReplayProvider:
    """Devuelve las respuestas guardadas en vez de llamar a un modelo.

    En clasificación y búsqueda filtra la respuesta a las páginas del bloque
    que pide el pipeline, como si el modelo solo hubiera visto ese bloque.
    """

    def __init__(self, answers: dict):
        self.answers = answers

    def generate(self, request: Request) -> LLMResponse:
        name = request.prompt_name
        if request.task == "classify":
            block = self._block(request)
            items = self.answers["classification"][name]["clasificaciones"]
            payload = {"clasificaciones": [i for i in items if i["page_number"] in block]}
        elif request.task == "search":
            block = self._block(request)
            stored = self.answers["search"].get(name, {"relevant_sections": [], "alerts": []})
            payload = {
                "relevant_sections": [
                    s for s in stored["relevant_sections"] if s["page_number"] in block
                ],
                "alerts": stored.get("alerts", []),
            }
        elif request.task == "identity":
            payload = self.answers["identity"]
        elif request.task == "extract":
            payload = self.answers["extraction"].get(name, {"filas": []})
        else:
            payload = self.answers["quality"].get(name, {"estado": "Aprobado", "observaciones": []})
        return LLMResponse(text=json.dumps(payload, ensure_ascii=False), finish_reason="STOP")

    @staticmethod
    def _block(request: Request) -> set[int]:
        return {
            int(m.group(1))
            for part in request.parts
            if part.text
            for m in [_PAGE_LABEL.search(part.text)]
            if m
        }


def run_pipeline(pdf: Path, answers: dict, out_dir: Path) -> dict:
    import os
    import tempfile

    os.environ["AI_PROVIDER"] = "fake"
    os.environ["STORAGE_DIR"] = tempfile.mkdtemp(prefix="eval_storage_")
    os.environ["TEMP_DIR"] = tempfile.mkdtemp(prefix="eval_temp_")

    from fastapi.testclient import TestClient

    from app.adapters import llm
    from app.config import get_settings

    get_settings.cache_clear()
    replay = ReplayProvider(answers)
    llm.get_provider = lambda *args, **kwargs: replay

    from app.main import app

    def wait(client, job):
        while True:
            state = client.get(f"/v1/jobs/{job['job_id']}").json()
            if state["status"] in {"succeeded", "failed", "cancelled"}:
                return state
            time.sleep(0.2)

    summary: dict = {}
    with TestClient(app) as client, pdf.open("rb") as handle:
        doc = client.post(
            "/v1/documents", files={"file": (pdf.name, handle, "application/pdf")}
        ).json()
        doc_id = doc["document_id"]
        plan = wait(client, client.post(f"/v1/documents/{doc_id}/plan", json={}).json())
        summary["plan"] = plan["result"] if plan["status"] == "succeeded" else plan["error"]

        for key in answers.get("extraction", {}):
            state = wait(
                client,
                client.post(f"/v1/documents/{doc_id}/pipeline", json={"analysis_type": key}).json(),
            )
            result = state.get("result") or {}
            quality = result.get("quality") or {}
            summary[key] = {
                "status": state["status"],
                "error": state.get("error"),
                "paginas": (result.get("search") or {}).get("relevant_pages"),
                "calidad": quality.get("status"),
                "reglas": [
                    f"{c['rule']} {c['period']}: {c['status']}" for c in quality.get("checks", [])
                ],
            }
            xlsx = client.get(f"/v1/documents/{doc_id}/extraction/{key}/xlsx")
            if xlsx.status_code == 200:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{key}.xlsx").write_bytes(xlsx.content)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("document", help="Carpeta dentro de evals/, por ejemplo goldfields_2024.")
    parser.add_argument("model", help="Archivo de respuestas sin extensión, por ejemplo gemini_v3.")
    parser.add_argument("--pipeline", type=Path, help="PDF para reproducir el pipeline completo.")
    args = parser.parse_args()

    folder = ROOT / args.document
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    answers = json.loads((folder / f"{args.model}.json").read_text(encoding="utf-8"))

    report = {"score": score(expected, answers)}
    if args.pipeline:
        report["pipeline"] = run_pipeline(args.pipeline, answers, folder / "out" / args.model)

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
