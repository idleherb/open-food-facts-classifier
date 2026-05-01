"""FastAPI app + lifespan for the classifier service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vorrat_classifier import __version__
from vorrat_classifier.api.classify import router as classify_router
from vorrat_classifier.config import Settings, get_settings
from vorrat_classifier.inference.runner import ClassifierRunner
from vorrat_classifier.schemas import HealthzResponse

log = logging.getLogger(__name__)


def _build_runner(settings: Settings) -> ClassifierRunner:
    """Construct the runner appropriate for the current settings.

    No model path ⇒ a tiny stub that always reports `is_loaded=False`
    and refuses to classify. This keeps the service deployable
    (healthz answers, container starts, k8s readiness probes work)
    even when no GGUF has been mounted yet — matches the
    walking-skeleton-first protocol.

    Model path set ⇒ the real LlamaCppRunner. Imported lazily so
    `import vorrat_classifier.main` doesn't pay the llama_cpp
    initialisation cost when running tests against the stub.
    """
    if settings.model_path is None:
        return _UnloadedStub()

    # Lazy import: tests against the stub never pay the llama_cpp
    # initialisation cost, and importers of this module that don't
    # have llama_cpp built locally still get a working healthz path.
    from vorrat_classifier.inference.llama_runner import LlamaCppRunner  # noqa: PLC0415

    return LlamaCppRunner(
        model_path=settings.model_path,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        max_output_tokens=settings.max_output_tokens,
    )


class _UnloadedStub:
    """Sentinel runner used when no model is configured."""

    @property
    def model_id(self) -> str:
        return "unloaded"

    @property
    def is_loaded(self) -> bool:
        return False

    def classify(self, req):  # type: ignore[no-untyped-def]
        # The /classify endpoint short-circuits on is_loaded=False
        # before ever calling this; we still raise so an accidental
        # bypass surfaces loudly rather than returning a fake answer.
        raise RuntimeError("classifier model not loaded")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    runner = _build_runner(settings)
    app.state.runner = runner
    app.state.settings = settings
    log.warning(
        "classifier-service started: channel=%s sha=%s model_loaded=%s model_id=%s",
        settings.build_channel,
        settings.build_sha,
        runner.is_loaded,
        runner.model_id,
    )
    try:
        yield
    finally:
        # Llama instance frees its mmap on GC — explicit teardown is
        # not required by llama-cpp-python and trying to call .close()
        # on it would error. Drop the reference and let the process
        # exit handle the rest.
        app.state.runner = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="vorrat-classifier",
        version=__version__,
        description=(
            "Local LLM-backed product-category classifier for Vorrat. "
            "Stage 2 of ADR-0031 — runs Qwen 2.5-7B-Instruct via "
            "llama-cpp-python and returns one of 16 stable category "
            "buckets, GBNF-grammar-constrained."
        ),
        lifespan=lifespan,
    )

    @app.get("/healthz", response_model=HealthzResponse)
    async def healthz() -> HealthzResponse:
        runner: ClassifierRunner | None = getattr(app.state, "runner", None)
        settings: Settings | None = getattr(app.state, "settings", None)
        # Settings should always be present once lifespan ran; fall
        # through to defaults if a test hits /healthz outside lifespan.
        channel = settings.build_channel if settings else "dev"
        commit = settings.build_sha if settings else "unknown"
        version = settings.build_date if settings else __version__
        return HealthzResponse(
            ok=True,
            channel=channel,
            version=version,
            commit=commit,
            model_loaded=bool(runner and runner.is_loaded),
        )

    app.include_router(classify_router)
    return app


app = create_app()
