"""Settings loading + env-prefix behaviour."""

from __future__ import annotations

import pytest

from vorrat_classifier.config import Settings, get_settings


def test_settings_default_has_no_model_path() -> None:
    s = Settings()
    assert s.model_path is None
    assert s.n_ctx == 4096
    assert s.n_threads is None
    assert s.build_channel == "dev"
    assert s.max_output_tokens == 24


def test_settings_reads_from_prefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VORRAT_CLASSIFIER_MODEL_PATH", "/models/qwen.gguf")
    monkeypatch.setenv("VORRAT_CLASSIFIER_N_CTX", "8192")
    monkeypatch.setenv("VORRAT_CLASSIFIER_BUILD_CHANNEL", "stable")
    monkeypatch.setenv("VORRAT_CLASSIFIER_BUILD_SHA", "abc1234")
    s = Settings()
    assert s.model_path == "/models/qwen.gguf"
    assert s.n_ctx == 8192
    assert s.build_channel == "stable"
    assert s.build_sha == "abc1234"


def test_settings_rejects_negative_n_ctx() -> None:
    """Pydantic validators must catch obvious wire-config bugs."""
    with pytest.raises(ValueError):
        Settings(n_ctx=-1)


def test_get_settings_returns_settings() -> None:
    """The lazy loader is the canonical entrypoint — alias for Settings()."""
    assert isinstance(get_settings(), Settings)
