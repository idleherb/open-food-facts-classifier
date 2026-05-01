"""Health endpoint behaviour."""

from __future__ import annotations

from httpx import AsyncClient


async def test_healthz_returns_ok_with_loaded_model(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["model_loaded"] is True
    assert "channel" in body
    assert "version" in body
    assert "commit" in body


async def test_healthz_reports_unloaded_model(unloaded_client: AsyncClient) -> None:
    """When the runner is unloaded, /healthz answers but flags it.

    This is the readiness-vs-liveness split: the service is alive
    (200) but not yet ready to classify (model_loaded=False). An
    orchestrator would probe model_loaded for readiness.
    """
    response = await unloaded_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
