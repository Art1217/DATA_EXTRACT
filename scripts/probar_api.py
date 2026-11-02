"""Recorrido manual de la API, etapa por etapa.

Lanza cada endpoint contra un servidor ya encendido y espera a que el job
termine antes de pasar al siguiente, imprimiendo el progreso. Sirve para
comprobar el backend de punta a punta con un documento real.

Con `AI_PROVIDER=fake` (el valor por defecto) no cuesta nada. Con `vertex` o
`gemini_api`, salvo `--solo-catalogo`, esto **llama al modelo y cuesta
dinero**: usa `--max-pages` para acotar el recorrido mientras pruebas.

Ejemplos
--------
    # Solo lo gratuito: catálogo, configuración
    python scripts/probar_api.py --solo-catalogo

    # Clasificación limitada a 5 páginas (barato)
    python scripts/probar_api.py data/eeff-2024.pdf --hasta plan --max-pages 5

    # Recorrido completo de un análisis
    python scripts/probar_api.py data/eeff-2024.pdf --analisis situacion_financiera
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ETAPAS = (
    "catalogo",
    "documento",
    "grupos",
    "analisis",
    "plan",
    "identidad",
    "busqueda",
    "extraccion",
    "calidad",
)


def imprimir(titulo: str) -> None:
    print(f"\n{'=' * 70}\n{titulo}\n{'=' * 70}")


def resumir(payload, limite: int = 600) -> str:
    texto = json.dumps(payload, indent=2, ensure_ascii=False)
    if len(texto) > limite:
        return texto[:limite] + f"\n  ... ({len(texto)} caracteres en total)"
    return texto


def esperar_job(cliente: httpx.Client, job_id: str, timeout: float = 1800.0) -> dict:
    """Sondea el job mostrando el progreso hasta que termina."""
    limite = time.monotonic() + timeout
    ultimo = ""
    while time.monotonic() < limite:
        job = cliente.get(f"/v1/jobs/{job_id}").raise_for_status().json()
        mensaje = f"  [{job['progress'] * 100:5.1f}%] {job['message']}"
        if mensaje != ultimo:
            print(mensaje)
            ultimo = mensaje
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(1.0)
    raise TimeoutError(f"El job {job_id} no terminó en {timeout} s")


def lanzar(cliente: httpx.Client, ruta: str, cuerpo: dict | None = None) -> dict:
    """POST asíncrono: encola, espera y devuelve el job terminado."""
    respuesta = cliente.post(ruta, json=cuerpo or {})
    if respuesta.status_code != 202:
        print(f"  ERROR {respuesta.status_code}: {respuesta.text}")
        respuesta.raise_for_status()
    job = esperar_job(cliente, respuesta.json()["job_id"])
    if job["status"] != "succeeded":
        print(f"  FALLÓ: {job['error']}")
    return job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", help="PDF a procesar.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--analisis", default="situacion_financiera", help="Clave del catálogo a procesar."
    )
    parser.add_argument("--grupo", default="Estados", help="Grupo para la Fase 2.")
    parser.add_argument("--max-pages", type=int, help="Limita el recorrido (abarata la prueba).")
    parser.add_argument("--api-key", help="Solo si el servidor tiene API_KEY configurada.")
    parser.add_argument(
        "--hasta",
        choices=ETAPAS,
        default="calidad",
        help="Última etapa a ejecutar. Por defecto, el recorrido completo.",
    )
    parser.add_argument(
        "--solo-catalogo",
        action="store_true",
        help="Solo endpoints que no llaman al modelo.",
    )
    args = parser.parse_args()

    cabeceras = {"X-API-Key": args.api_key} if args.api_key else {}
    hasta = ETAPAS.index("catalogo") if args.solo_catalogo else ETAPAS.index(args.hasta)
    recorrido = {"max_pages": args.max_pages} if args.max_pages else {}

    with httpx.Client(base_url=args.url, headers=cabeceras, timeout=60.0) as cliente:
        # --- Comprobaciones gratuitas -------------------------------------
        imprimir("1. Salud y configuración")
        try:
            print("  healthz:", cliente.get("/healthz").raise_for_status().json())
        except httpx.ConnectError:
            print(f"  No hay servidor en {args.url}.")
            print("  Arráncalo con:  python -m uvicorn app.main:app --port 8000")
            return 1
        print("  config:", resumir(cliente.get("/v1/config").json(), 300))

        imprimir("2. Catálogo (lo que el backend sabe hacer)")
        catalogo = cliente.get("/v1/catalog/analysis-types").json()
        print(f"  {len(catalogo)} tipos de análisis")
        for grupo in cliente.get("/v1/catalog/groups").json():
            print(f"    {grupo['group']:8} {', '.join(grupo['codes'])}")

        if hasta <= ETAPAS.index("catalogo"):
            print("\nListo (solo catálogo).")
            return 0

        if not args.pdf:
            parser.error("hace falta un PDF salvo con --solo-catalogo")
        ruta_pdf = Path(args.pdf)
        if not ruta_pdf.is_file():
            print(f"No existe el archivo: {ruta_pdf}")
            return 1

        # --- Alta del documento (tampoco usa el modelo) -------------------
        imprimir("3. Alta del documento")
        with ruta_pdf.open("rb") as handle:
            documento = (
                cliente.post(
                    "/v1/documents",
                    files={"file": (ruta_pdf.name, handle, "application/pdf")},
                )
                .raise_for_status()
                .json()
            )
        document_id = documento["document_id"]
        print(f"  document_id: {document_id}")
        print(f"  {documento['pages']} páginas | escaneado: {documento['is_scanned']}")

        if hasta <= ETAPAS.index("documento"):
            return 0

        if cliente.get("/v1/config").json().get("ai_provider") != "fake":
            print("\n  A partir de aquí se llama al modelo y se incurre en costo.")

        # --- Fase 1 -------------------------------------------------------
        imprimir("4. Fase 1 — clasificación de grupos")
        job = lanzar(cliente, f"/v1/documents/{document_id}/classification/groups", recorrido)
        if job["status"] == "succeeded":
            print("  grupos:", job["result"]["groups"])

        if hasta <= ETAPAS.index("grupos"):
            return 0

        # --- Fase 2 -------------------------------------------------------
        imprimir(f"5. Fase 2 — análisis dentro de {args.grupo}")
        job = lanzar(
            cliente,
            f"/v1/documents/{document_id}/classification/analysis",
            recorrido | {"group": args.grupo},
        )
        if job["status"] == "succeeded":
            print("  análisis:", job["result"]["analyses"])
            print("  claves:  ", job["result"]["analysis_types"])

        if hasta <= ETAPAS.index("analisis"):
            return 0

        # --- Plan ---------------------------------------------------------
        imprimir("6. Plan (reutiliza las dos fases anteriores)")
        job = lanzar(cliente, f"/v1/documents/{document_id}/plan", recorrido)
        if job["status"] == "succeeded":
            print("  plan:", job["result"]["groups"])

        if hasta <= ETAPAS.index("plan"):
            return 0

        # --- Identidad ----------------------------------------------------
        imprimir("7. Identidad del documento")
        job = lanzar(cliente, f"/v1/documents/{document_id}/identity")
        if job["status"] == "succeeded":
            print(resumir(job["result"], 400))

        if hasta <= ETAPAS.index("identidad"):
            return 0

        # --- Búsqueda -----------------------------------------------------
        imprimir(f"8. Búsqueda de páginas — {args.analisis}")
        job = lanzar(
            cliente,
            f"/v1/documents/{document_id}/search",
            recorrido | {"analysis_type": args.analisis},
        )
        if job["status"] == "succeeded":
            print("  páginas relevantes:", job["result"]["relevant_pages"])
            print("  bloques fallidos:  ", job["result"]["blocks_failed"])
            print(f"  PDF recortado: GET /v1/documents/{document_id}/search/{args.analisis}/pdf")

        if hasta <= ETAPAS.index("busqueda"):
            return 0

        # --- Extracción ---------------------------------------------------
        imprimir(f"9. Extracción — {args.analisis}  (la etapa cara)")
        job = lanzar(
            cliente,
            f"/v1/documents/{document_id}/extraction",
            {"analysis_type": args.analisis},
        )
        if job["status"] == "succeeded":
            print("  filas extraídas:", job["result"]["row_count"])
            if job["result"]["rows"]:
                print("  primera fila:", resumir(job["result"]["rows"][0], 400))
            print(f"  Excel: GET /v1/documents/{document_id}/extraction/{args.analisis}/xlsx")

        if hasta <= ETAPAS.index("extraccion"):
            return 0

        # --- Calidad ------------------------------------------------------
        imprimir(f"10. Control de calidad — {args.analisis}")
        job = lanzar(
            cliente,
            f"/v1/documents/{document_id}/quality",
            {"analysis_type": args.analisis},
        )
        if job["status"] == "succeeded":
            print("  estado:", job["result"]["status"])
            for regla in job["result"]["checks"]:
                print(f"    [{regla['status']:>7}] {regla['period']}  {regla['description']}")
            for error in job["result"]["errors"]:
                print("  observación:", error)

        # --- Resumen ------------------------------------------------------
        imprimir("Resumen")
        ficha = cliente.get(f"/v1/documents/{document_id}").json()
        print("  etapas ejecutadas:", resumir(ficha["stages"], 500))
        print(f"\n  Documento: {document_id}")
        print(f"  Artefactos en: Storage/documents/{document_id}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
