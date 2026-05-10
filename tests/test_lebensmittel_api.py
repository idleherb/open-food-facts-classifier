"""Integration tests for POST /lebensmittel."""

from __future__ import annotations

from httpx import AsyncClient


async def test_lebensmittel_returns_runner_id_for_loaded_runner(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/lebensmittel",
        json={"name": "Spaghetti No.5", "brand": "Barilla"},
    )
    assert response.status_code == 200
    body = response.json()
    # StubRunner returns its configured fixed id.
    assert body["lebensmittel_id"] == "vorrat:stub"
    assert body["model_id"] == "stub"
    assert body["inference_ms"] == 1


async def test_lebensmittel_returns_503_when_runner_unloaded(
    unloaded_client: AsyncClient,
) -> None:
    response = await unloaded_client.post("/lebensmittel", json={"name": "anything"})
    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"]


async def test_lebensmittel_validates_required_name(client: AsyncClient) -> None:
    response = await client.post("/lebensmittel", json={"brand": "no name"})
    assert response.status_code == 422


async def test_lebensmittel_accepts_off_categories_tags(client: AsyncClient) -> None:
    """Optional fields plumb through without 422."""
    response = await client.post(
        "/lebensmittel",
        json={
            "name": "Bio Fusilli Vollkorn",
            "brand": "Alnatura",
            "generic_name": "Hartweizenpasta vollkorn",
            "off_categories_tags": [
                "en:cereals",
                "en:pastas",
                "en:wholegrain-pastas",
                "en:fusilli",
            ],
        },
    )
    assert response.status_code == 200


async def test_lebensmittel_rejects_unknown_fields(client: AsyncClient) -> None:
    """`extra='forbid'` on LebensmittelRequest catches typos."""
    response = await client.post(
        "/lebensmittel",
        json={"name": "X", "not_a_real_field": "value"},
    )
    assert response.status_code == 422
