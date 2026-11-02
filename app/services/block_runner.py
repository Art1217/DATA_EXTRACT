"""Recorrido de un PDF por bloques de páginas, en paralelo y por tandas.

La búsqueda y las dos fases de clasificación usan el mismo algoritmo con
distinta función de destino, así que aquí queda una sola vez y recibe esa
función como parámetro.

El patrón es "renderizar justo a tiempo": se generan las imágenes de una tanda
de bloques, se envían en paralelo, y se borran en cuanto la tanda termina. Un
documento de 500 páginas a 300 DPI no cabe en disco si se renderiza entero de
golpe.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from app.adapters.preprocessing import preprocess_pdf_page
from app.services.context import CancellationToken, Progress
from app.services.limits import llm_slot

logger = logging.getLogger(__name__)

PageImage = tuple[int, str]
"""(número de página 1-indexado, ruta de la imagen renderizada)."""

BlockCall = Callable[[Sequence[PageImage]], Any]


@dataclass(frozen=True, slots=True)
class BlockOutcome:
    """Resultado de un bloque. Un fallo no aborta el recorrido completo."""

    index: int
    first_page: int
    last_page: int
    value: Any | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def run_in_blocks(
    pdf_path: Path | str,
    call: BlockCall,
    *,
    workspace_dir: Path,
    block_size: int,
    workers: int,
    dpi: int,
    progress: Progress,
    cancellation: CancellationToken,
    max_pages: int | None = None,
    label: str = "Bloque",
) -> list[BlockOutcome]:
    """Aplica `call` a cada bloque de páginas del documento.

    Devuelve un resultado por bloque, ordenado por índice. Los bloques que
    fallan se marcan con `error` en vez de propagar la excepción: perder un
    bloque de treinta páginas no justifica descartar el trabajo de los otros
    veinte bloques que sí respondieron.
    """
    document = pymupdf.open(str(pdf_path))
    try:
        total_pages = document.page_count
        if max_pages:
            total_pages = min(total_pages, max_pages)

        block_starts = list(range(0, total_pages, block_size))
        if not block_starts:
            return []

        outcomes: list[BlockOutcome] = []
        completed = 0

        for batch_start in range(0, len(block_starts), workers):
            cancellation.raise_if_cancelled()
            batch = block_starts[batch_start : batch_start + workers]

            progress.report(
                f"Preparando páginas {batch[0] + 1}-{min(batch[-1] + block_size, total_pages)}...",
                completed / len(block_starts),
            )
            rendered = [
                _render_block(
                    document,
                    index=batch_start + offset,
                    start_page=start,
                    end_page=min(start + block_size, total_pages),
                    dpi=dpi,
                    workspace_dir=workspace_dir,
                )
                for offset, start in enumerate(batch)
            ]

            try:
                outcomes.extend(_dispatch(rendered, call, workers))
                completed += len(batch)
                progress.report(
                    f"{label} {completed}/{len(block_starts)} completado.",
                    completed / len(block_starts),
                )
            finally:
                _discard_images(rendered)

        return sorted(outcomes, key=lambda outcome: outcome.index)
    finally:
        document.close()


@dataclass(frozen=True, slots=True)
class _RenderedBlock:
    index: int
    first_page: int
    last_page: int
    images: list[PageImage]


def _render_block(
    document: pymupdf.Document,
    *,
    index: int,
    start_page: int,
    end_page: int,
    dpi: int,
    workspace_dir: Path,
) -> _RenderedBlock:
    """Renderiza las páginas de un bloque, saltando las que fallen."""
    images: list[PageImage] = []
    for page_number in range(start_page, end_page):
        try:
            path = preprocess_pdf_page(document, page_number, dpi=dpi, out_dir=workspace_dir)
            images.append((page_number + 1, path))
        except Exception as exc:
            logger.warning("Página %d no pudo renderizarse: %s. Se omite.", page_number + 1, exc)
    return _RenderedBlock(index, start_page + 1, end_page, images)


def _dispatch(
    blocks: Sequence[_RenderedBlock], call: BlockCall, workers: int
) -> list[BlockOutcome]:
    """Envía una tanda de bloques en paralelo y espera a que terminen todos."""

    def invoke(block: _RenderedBlock) -> BlockOutcome:
        try:
            with llm_slot():
                return BlockOutcome(
                    block.index, block.first_page, block.last_page, value=call(block.images)
                )
        except Exception as exc:
            logger.warning(
                "Bloque %d (páginas %d-%d) falló: %s",
                block.index + 1,
                block.first_page,
                block.last_page,
                _describe(exc),
            )
            return BlockOutcome(
                block.index, block.first_page, block.last_page, error=_describe(exc)
            )

    results: list[BlockOutcome] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # `copy_context` propaga el contexto de trazado de LangSmith al hilo
        # trabajador; sin esto cada bloque aparecería como una traza suelta.
        futures = [
            executor.submit(contextvars.copy_context().run, invoke, block) for block in blocks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _discard_images(blocks: Sequence[_RenderedBlock]) -> None:
    for block in blocks:
        for _, path in block.images:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("No se pudo borrar la imagen temporal %s: %s", path, exc)


def _describe(exc: Exception) -> str:
    """Mensaje legible, desenvolviendo el envoltorio de reintentos de tenacity."""
    last_attempt = getattr(exc, "last_attempt", None)
    if last_attempt is not None:
        cause = last_attempt.exception()
        if cause is not None:
            return f"{type(cause).__name__}: {cause}"
    return f"{type(exc).__name__}: {exc}"
