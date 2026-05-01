"""Lifespan-level behaviour: stub-fallback, HF-download path, state setup."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from off_classifier.config import Settings
from off_classifier.main import (
    _build_runner,
    _resolve_model_path,
    _UnloadedStub,
    app,
    create_app,
)


async def test_lifespan_attaches_unloaded_stub_with_test_defaults() -> None:
    """conftest sets OFF_CLASSIFIER_MODEL_REPO="" so the suite never hits
    the network — the default lifespan therefore lands on the stub."""
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


async def test_build_runner_returns_stub_when_repo_disabled() -> None:
    """Empty model_repo + no override ⇒ explicit opt-out of any download."""
    runner = _build_runner(Settings(model_repo="", model_path_override=None))
    assert isinstance(runner, _UnloadedStub)


async def test_build_runner_returns_stub_when_override_path_missing() -> None:
    """Override is set but the file doesn't exist — operator bug, fall to stub."""
    runner = _build_runner(
        Settings(
            model_repo="",
            model_path_override="/nonexistent/no-such-file.gguf",
        )
    )
    assert isinstance(runner, _UnloadedStub)


async def test_resolve_model_path_uses_override_when_file_exists(
    tmp_path: Path,
) -> None:
    """Override path with a real file should be returned verbatim,
    short-circuiting any HF download."""
    fake = tmp_path / "fake.gguf"
    fake.write_bytes(b"not really a gguf")

    settings = Settings(
        model_repo="bartowski/some-other-repo",  # would be wrong if downloaded
        model_path_override=str(fake),
    )
    assert _resolve_model_path(settings) == str(fake)


async def test_resolve_model_path_calls_hf_hub_download(tmp_path: Path) -> None:
    """When repo is set + no override, the resolver delegates to HF and
    returns the path that hf_hub_download produced."""
    expected = tmp_path / "downloaded.gguf"
    expected.write_bytes(b"x")

    settings = Settings(
        model_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        model_filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        model_path_override=None,
    )
    with patch("huggingface_hub.hf_hub_download", return_value=str(expected)) as mock_dl:
        result = _resolve_model_path(settings)
    assert result == str(expected)
    mock_dl.assert_called_once_with(
        repo_id="bartowski/Qwen2.5-7B-Instruct-GGUF",
        filename="Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    )


async def test_resolve_model_path_returns_none_on_hf_oserror() -> None:
    """Network/DNS/offline failures must not crash the container —
    they fall to the stub so the service degrades silently."""
    settings = Settings(
        model_repo="bartowski/Qwen2.5-7B-Instruct-GGUF",
        model_path_override=None,
    )
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=OSError("connection refused"),
    ):
        assert _resolve_model_path(settings) is None


async def test_resolve_model_path_returns_none_on_hf_http_error() -> None:
    """A 404/401/403 from HF (wrong repo, bad token, gated model) is
    operator-config bug, not a runtime crash — fall to stub."""
    import httpx  # noqa: PLC0415
    from huggingface_hub.errors import HfHubHTTPError  # noqa: PLC0415

    settings = Settings(
        model_repo="nonexistent/wrong-repo",
        model_path_override=None,
    )
    response = httpx.Response(
        404,
        request=httpx.Request("GET", "https://huggingface.co/api/x"),
        text="Repository not found",
    )
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=HfHubHTTPError("Repository not found", response=response),
    ):
        assert _resolve_model_path(settings) is None


async def test_resolve_model_path_returns_none_on_value_error() -> None:
    """huggingface_hub raises ValueError for malformed repo IDs etc.
    Our resolver catches that as a configuration bug and degrades."""
    settings = Settings(
        model_repo="malformed///id",
        model_path_override=None,
    )
    with patch(
        "huggingface_hub.hf_hub_download",
        side_effect=ValueError("Invalid repo_id"),
    ):
        assert _resolve_model_path(settings) is None


async def test_healthz_works_during_lifespan_with_default_settings() -> None:
    """Walking-skeleton bar: a deployment without a downloaded GGUF
    must still answer /healthz. Ops needs this for readiness probes."""
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
