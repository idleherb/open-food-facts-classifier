"""Shared fixtures.

Tests run against a deterministic stub runner — no GGUF on disk,
no llama_cpp import. The only test that touches a real model is the
explicit smoke test under `tests/test_smoke_real_model.py`, which is
skipped unless `VORRAT_CLASSIFIER_SMOKE_MODEL_PATH` is set.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from vorrat_classifier.api.classify import get_runner
from vorrat_classifier.inference.runner import ClassifierRunner
from vorrat_classifier.main import app
from vorrat_classifier.schemas import ClassifyRequest, ClassifyResponse
from vorrat_classifier.taxonomy import Category


class StubRunner:
    """Deterministic runner: returns a fixed category, regardless of input.

    Configurable via `category` so tests can verify the response is
    actually plumbed through (vs e.g. always returning 'sonstiges').
    """

    def __init__(
        self,
        *,
        category: Category = "sonstiges",
        is_loaded: bool = True,
        model_id: str = "stub",
        inference_ms: int = 1,
    ) -> None:
        self._category = category
        self._is_loaded = is_loaded
        self._model_id = model_id
        self._inference_ms = inference_ms
        self.calls: list[ClassifyRequest] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def classify(self, req: ClassifyRequest) -> ClassifyResponse:
        self.calls.append(req)
        return ClassifyResponse(
            category=self._category,
            model_id=self._model_id,
            inference_ms=self._inference_ms,
        )


@pytest.fixture
def stub_runner() -> StubRunner:
    return StubRunner()


async def _make_client(runner: StubRunner) -> AsyncIterator[AsyncClient]:
    """Wire `runner` into the app via dependency override + lifespan."""

    def _override() -> ClassifierRunner:
        return runner

    app.dependency_overrides[get_runner] = _override
    try:
        async with LifespanManager(app):
            # Lifespan attached its own (unloaded) runner, but we
            # also want /healthz to reflect the stub. Swap it.
            app.state.runner = runner
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as c:
                yield c
    finally:
        app.dependency_overrides.pop(get_runner, None)


@pytest.fixture
async def client(stub_runner: StubRunner) -> AsyncIterator[AsyncClient]:
    async for c in _make_client(stub_runner):
        yield c


@pytest.fixture
def unloaded_stub_runner() -> StubRunner:
    return StubRunner(is_loaded=False, model_id="unloaded")


@pytest.fixture
async def unloaded_client(
    unloaded_stub_runner: StubRunner,
) -> AsyncIterator[AsyncClient]:
    async for c in _make_client(unloaded_stub_runner):
        yield c
