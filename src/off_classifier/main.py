"""FastAPI app + lifespan for the classifier service."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from off_classifier import __version__
from off_classifier.api.classify import router as classify_router
from off_classifier.api.lebensmittel import router as lebensmittel_router
from off_classifier.config import Settings, get_settings
from off_classifier.inference.runner import ClassifierRunner
from off_classifier.schemas import HealthzResponse

log = logging.getLogger(__name__)


def _resolve_model_path(settings: Settings) -> str | None:
    """Locate the GGUF on disk, downloading from HuggingFace if needed.

    Resolution order:
      1. ``model_path_override`` set + file exists ⇒ use it directly.
      2. ``model_path_override`` set + file *missing* ⇒ stub (and warn).
         The override semantically means "skip auto-download", so a
         missing file is the operator's bug, not ours to paper over.
      3. ``model_repo`` empty ⇒ stub. The operator explicitly disabled
         auto-download without providing an override.
      4. Otherwise: ``hf_hub_download(model_repo, model_filename)``.
         HuggingFace handles caching via HF_HOME (set in the Dockerfile
         to ``/models/hf_cache``, backed by a named volume), so the
         second container start hits the cache and is fast. HF_TOKEN
         is read by ``huggingface_hub`` from the env automatically.

    Any exception during download is caught and logged; we fall to the
    stub rather than crashing the container at startup. Watchtower
    rolling forward to a known-bad classifier release shouldn't take
    the service offline; it should degrade silently to "OFF-only" on
    the vorrat side.
    """
    from pathlib import Path  # noqa: PLC0415

    if settings.model_path_override:
        path = Path(settings.model_path_override)
        if path.is_file():
            return str(path)
        log.warning(
            "model_path_override %s is set but no file there; running as stub",
            settings.model_path_override,
        )
        return None

    if not settings.model_repo:
        log.warning(
            "model_repo empty and no model_path_override set; running as stub",
        )
        return None

    # Lazy import — keeps the unit-test suite import-free of huggingface_hub
    # internals and lets a stub-only container omit the dep download too.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415

    try:
        return hf_hub_download(
            repo_id=settings.model_repo,
            filename=settings.model_filename,
        )
    except (HfHubHTTPError, OSError, ValueError) as exc:
        # HfHubHTTPError covers 404 (wrong repo/file), 401 (token bad),
        # 403 (gated), 429, 5xx. OSError covers offline / DNS failures.
        # ValueError covers malformed config. Everything else propagates.
        log.warning(
            "HF download of %s/%s failed (%s); running as stub",
            settings.model_repo,
            settings.model_filename,
            exc,
        )
        return None


def _build_runner(settings: Settings) -> ClassifierRunner:
    """Resolve a GGUF (or fall to stub) and wrap it in a runner.

    The file-resolution dance lives in ``_resolve_model_path``; this
    function only decides "stub vs llama-cpp" once the path question
    is settled. Lazy-imports LlamaCppRunner so test suites that never
    touch the real path don't pay the llama_cpp init cost.
    """
    path = _resolve_model_path(settings)
    if path is None:
        return _UnloadedStub()

    from off_classifier.inference.llama_runner import LlamaCppRunner  # noqa: PLC0415

    return LlamaCppRunner(
        model_path=path,
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

    def lebensmittel(self, req):  # type: ignore[no-untyped-def]
        # Same short-circuit pattern as classify — /lebensmittel
        # checks is_loaded and 503s before reaching this method.
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
        title="open-food-facts-classifier",
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
    app.include_router(lebensmittel_router)
    return app


app = create_app()
