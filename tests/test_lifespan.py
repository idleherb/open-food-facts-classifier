"""Lifespan-level behaviour: the unloaded-stub branch + state setup."""

from __future__ import annotations

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from off_classifier.config import Settings
from off_classifier.main import _build_runner, _UnloadedStub, app, create_app


async def test_lifespan_attaches_unloaded_stub_when_no_model_path() -> None:
    """Default settings ⇒ no model path ⇒ stub is attached."""
    async with LifespanManager(app):
        runner = app.state.runner
        assert isinstance(runner, _UnloadedStub)
        assert runner.is_loaded is False
        assert runner.model_id == "unloaded"


async def test_unloaded_stub_classify_raises() -> None:
    """The stub is never supposed to be invoked — the API short-circuits
    on is_loaded=False — but if something bypasses that, fail loud."""
    stub = _UnloadedStub()
    with pytest.raises(RuntimeError, match="not loaded"):
        stub.classify(None)


async def test_build_runner_returns_stub_for_no_model_path() -> None:
    runner = _build_runner(Settings(model_path=None))
    assert isinstance(runner, _UnloadedStub)


async def test_build_runner_returns_stub_when_model_file_missing() -> None:
    """Container default sets MODEL_PATH=/models/...gguf — without a
    volume mount that path doesn't exist. The lifespan must fall to
    the stub and warn, not crash the app at startup.
    """
    runner = _build_runner(Settings(model_path="/nonexistent/no-such-file.gguf"))
    assert isinstance(runner, _UnloadedStub)


async def test_healthz_works_during_lifespan_with_default_settings() -> None:
    """Walking-skeleton bar: a deployment with no GGUF mounted must
    still answer /healthz. Ops needs this for readiness probes."""
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.get("/healthz")
            assert response.status_code == 200
            body = response.json()
            assert body["ok"] is True
            assert body["model_loaded"] is False


async def test_create_app_returns_independent_instance() -> None:
    """`create_app()` must produce a fresh FastAPI each call so tests
    can isolate per-app dependency overrides without touching the
    module-level singleton."""
    a = create_app()
    b = create_app()
    assert a is not b
