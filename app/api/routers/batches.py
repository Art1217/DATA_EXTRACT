"""Corridas ya procesadas: qué se ejecutó y cómo volver a bajarlo.

Un lote no es una entidad del backend, es una etiqueta que viaja en cada job
(ver `app/services/batches.py`). Estos endpoints la leen al revés, para que la
interfaz pueda ofrecer "elige una corrida" sin depender de que la pestaña del
navegador siga abierta desde que se procesó.

El ZIP no se sirve aquí: sigue en `/v1/bundles/{batch_id}`, que es donde se
empaqueta y se descarga.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import JobStoreDep, StorageDep
from app.services import batches

router = APIRouter(prefix="/v1/batches", tags=["corridas"])


@router.get("")
def list_batches(
    store: JobStoreDep, storage: StorageDep, limit: int = Query(default=50, ge=1, le=500)
) -> list[dict]:
    """Corridas conocidas, de la más reciente a la más antigua."""
    return batches.list_batches(store, storage, limit=limit)


@router.get("/{batch_id}")
def get_batch(batch_id: str, store: JobStoreDep, storage: StorageDep) -> dict:
    """Documentos de una corrida y el desenlace de cada uno.

    No devuelve 404 para una corrida desconocida: devuelve una corrida vacía.
    Que no haya jobs con esa etiqueta no es un error del cliente, y quien
    pregunta quiere saber qué hay, no si acertó el nombre.
    """
    return batches.batch_detail(store, storage, batch_id)
