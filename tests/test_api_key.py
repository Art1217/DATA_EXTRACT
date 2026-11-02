"""Clave de acceso a la API.

No confundir con las credenciales de Google Cloud: aquélla controla quién
puede llamar a *este* backend, y las de Google son las que el backend usa por
su cuenta para hablar con Vertex AI y Document AI. Son independientes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_con_clave(settings, fake_ai, monkeypatch):
    """Servidor con `API_KEY` configurada."""
    from app.config import get_settings

    monkeypatch.setenv("API_KEY", "secreto-de-prueba")
    get_settings.cache_clear()
    get_settings().ensure_directories()

    from app.main import app

    with TestClient(app) as cliente:
        yield cliente

    get_settings.cache_clear()


def test_sin_clave_configurada_la_api_queda_abierta(client):
    """Es lo razonable en una instalación local dentro de una red privada."""
    assert client.get("/v1/catalog/groups").status_code == 200
    assert client.get("/v1/documents").status_code == 200


def test_con_clave_configurada_se_rechaza_a_quien_no_la_manda(client_con_clave):
    respuesta = client_con_clave.get("/v1/catalog/groups")
    assert respuesta.status_code == 401


def test_con_clave_correcta_se_permite(client_con_clave):
    respuesta = client_con_clave.get(
        "/v1/catalog/groups", headers={"X-API-Key": "secreto-de-prueba"}
    )
    assert respuesta.status_code == 200


def test_una_clave_equivocada_se_rechaza(client_con_clave):
    respuesta = client_con_clave.get("/v1/catalog/groups", headers={"X-API-Key": "otra-cosa"})
    assert respuesta.status_code == 401


def test_healthz_no_exige_clave(client_con_clave):
    """Un chequeo de salud debe responder sin credenciales."""
    assert client_con_clave.get("/healthz").status_code == 200


def test_la_clave_se_publica_como_esquema_de_seguridad(client):
    """En Swagger debe salir como un botón "Authorize", no como un campo
    de texto repetido en cada endpoint."""
    esquema = client.get("/openapi.json").json()
    seguridad = esquema["components"]["securitySchemes"]
    assert "APIKeyHeader" in seguridad
    assert seguridad["APIKeyHeader"]["name"] == "X-API-Key"
    assert seguridad["APIKeyHeader"]["in"] == "header"


def test_la_cabecera_ya_no_aparece_como_parametro_de_cada_endpoint(client):
    esquema = client.get("/openapi.json").json()
    parametros = esquema["paths"]["/v1/catalog/groups"]["get"].get("parameters", [])
    assert not any(p.get("name") == "X-API-Key" for p in parametros)
