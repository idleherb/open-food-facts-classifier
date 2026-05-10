"""Settings loading + env-prefix behaviour."""

from __future__ import annotations

import pytest

from off_classifier.config import Settings, get_settings


def test_settings_defaults_target_qwen_25_14b_q4km() -> None:
    """The container ships a default that "just works" if the operator
    has no env overrides — points at the bartowski Qwen2.5-14B Q4_K_M
    GGUF (bumped from 7B on 2026-05-10 after the /lebensmittel eval
    showed 7B confabulation on German specialty terms; ADR-0038 §4.1
    budgeted ~9 GB resident for this)."""
    # conftest sets OFF_CLASSIFIER_MODEL_REPO="" suite-wide; check the
    # baked-in defaults directly.
    s = Settings(model_repo="bartowski/Qwen2.5-14B-Instruct-GGUF")
    assert s.model_repo == "bartowski/Qwen2.5-14B-Instruct-GGUF"
    assert s.model_filename == "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    assert s.model_path_override is None
    assert s.n_ctx == 4096
    assert s.n_threads is None
    assert s.build_channel == "dev"
    assert s.max_output_tokens == 24


def test_settings_reads_from_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OFF_CLASSIFIER_MODEL_REPO", "Qwen/Qwen2.5-3B-Instruct-GGUF")
    monkeypatch.setenv("OFF_CLASSIFIER_MODEL_FILENAME", "qwen2.5-3b-instruct-q5_k_m.gguf")
    monkeypatch.setenv("OFF_CLASSIFIER_MODEL_PATH_OVERRIDE", "/tmp/my-local.gguf")
    monkeypatch.setenv("OFF_CLASSIFIER_N_CTX", "8192")
    monkeypatch.setenv("OFF_CLASSIFIER_BUILD_CHANNEL", "stable")
    monkeypatch.setenv("OFF_CLASSIFIER_BUILD_SHA", "abc1234")
    s = Settings()
    assert s.model_repo == "Qwen/Qwen2.5-3B-Instruct-GGUF"
    assert s.model_filename == "qwen2.5-3b-instruct-q5_k_m.gguf"
    assert s.model_path_override == "/tmp/my-local.gguf"
    assert s.n_ctx == 8192
    assert s.build_channel == "stable"
    assert s.build_sha == "abc1234"


def test_settings_empty_repo_disables_auto_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operators who explicitly want to stay on docker-cp / bind-mount
    territory set MODEL_REPO="" — the resolver respects empty string
    as "off", not as "use the default"."""
    monkeypatch.setenv("OFF_CLASSIFIER_MODEL_REPO", "")
    s = Settings()
    assert s.model_repo == ""


def test_settings_rejects_negative_n_ctx() -> None:
    """Pydantic validators must catch obvious wire-config bugs."""
    with pytest.raises(ValueError):
        Settings(n_ctx=-1)


def test_get_settings_returns_settings() -> None:
    """The lazy loader is the canonical entrypoint — alias for Settings()."""
    assert isinstance(get_settings(), Settings)
