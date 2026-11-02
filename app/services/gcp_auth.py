"""Credenciales de Google Cloud para los clientes de Storage y BigQuery.

Un solo lugar decide de dónde salen las credenciales y qué proyecto se usa,
porque el bucket y el dataset tienen que coincidir en ambas cosas.

Los imports de `google.*` son perezosos a propósito: el backend arranca y los
tests corren sin `google-cloud-storage` ni `google-cloud-bigquery` instalados
mientras no se use GCS ni se publique.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def credentials_and_project(
    credentials_path: Path | None, project: str | None
) -> tuple[Any, str | None]:
    """Devuelve `(credenciales, proyecto)`.

    Con `credentials_path` se usa esa service account; sin ella, las
    Application Default Credentials (`gcloud auth application-default login`
    en local, la cuenta del servicio en Cloud Run). El proyecto declarado
    manda; si no lo hay, se toma el de las credenciales.
    """
    if credentials_path:
        from google.oauth2 import service_account

        credentials = service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=_SCOPES
        )
        return credentials, project or credentials.project_id

    import google.auth

    credentials, default_project = google.auth.default(scopes=_SCOPES)
    return credentials, project or default_project
