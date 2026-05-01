"""HTTP-level tests for POST /classify, with a stub runner."""

from __future__ import annotations

from httpx import ASGITransport, AsyncClient

from off_classifier.main import create_app
from tests.conftest import StubRunner


async def test_classify_happy_path(client: AsyncClient, stub_runner: StubRunner) -> None:
    """The runner is invoked exactly once and its response is plumbed
    through to the wire intact.
    """
    response = await client.post(
        "/classify",
        json={"name": "Mehl Type 405", "brand": "Aurora"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "sonstiges"  # the default stub category
    assert body["model_id"] == "stub"
    assert body["inference_ms"] >= 0

    # Stub captured the request — verify the prompt builder will see
    # both fields, not e.g. just the name.
    assert len(stub_runner.calls) == 1
    captured = stub_runner.calls[0]
    assert captured.name == "Mehl Type 405"
    assert captured.brand == "Aurora"


async def test_classify_returns_503_when_model_unloaded(
    unloaded_client: AsyncClient, unloaded_stub_runner: StubRunner
) -> None:
    """A request to /classify when the runner reports is_loaded=False
    must return 503, not crash, and not invoke the runner."""
    response = await unloaded_client.post("/classify", json={"name": "Anything"})
    assert response.status_code == 503
    assert "model not loaded" in response.json()["detail"]
    # Runner was NOT invoked — the loaded-check short-circuits.
    assert unloaded_stub_runner.calls == []


async def test_classify_returns_503_when_no_runner_attached() -> None:
    """If lifespan failed to attach a runner, /classify returns 503.

    Reproduces this state by constructing a fresh app without running
    its lifespan; app.state.runner is missing entirely, the
    dependency function falls into its 503 branch.
    """
    fresh = create_app()
    transport = ASGITransport(app=fresh)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        response = await c.post("/classify", json={"name": "Any"})
        assert response.status_code == 503
        assert "runner not initialised" in response.json()["detail"]


async def test_classify_rejects_empty_name(client: AsyncClient) -> None:
    response = await client.post("/classify", json={"name": ""})
    assert response.status_code == 422


async def test_classify_rejects_unknown_field(client: AsyncClient) -> None:
    response = await client.post(
        "/classify",
        json={"name": "Mehl", "made_up": "x"},
    )
    assert response.status_code == 422


async def test_classify_accepts_full_request_payload(
    client: AsyncClient, stub_runner: StubRunner
) -> None:
    """All optional fields wire through to the runner."""
    payload = {
        "name": "Hafermilch Barista",
        "brand": "Oatly",
        "generic_name": "Pflanzendrink",
        "description": "Haferdrink mit Säurregulator und Speisesalz",
        "off_categories_tags": ["en:beverages", "en:plant-based-milks"],
    }
    response = await client.post("/classify", json=payload)
    assert response.status_code == 200

    captured = stub_runner.calls[0]
    assert captured.brand == "Oatly"
    assert captured.generic_name == "Pflanzendrink"
    assert captured.description and "Säurregulator" in captured.description
    assert captured.off_categories_tags == ["en:beverages", "en:plant-based-milks"]
