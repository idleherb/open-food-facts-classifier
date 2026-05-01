"""Runtime settings, loaded from environment variables.

Defaults assume the production layout: model file mounted into
`/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` by Docker volume.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OFF_CLASSIFIER_",
        case_sensitive=False,
        extra="ignore",
    )

    # Path to the GGUF on disk. None ⇒ /classify returns a structured
    # 503 ("model not loaded"); the rest of the API surface still works
    # so deployments + healthz can be verified before the model is in
    # place. This matches the walking-skeleton-first protocol.
    model_path: str | None = None

    # Context window. Qwen 2.5 supports 128k natively; we only feed
    # ~200 tokens of prompt + ~10 tokens of output, so 4k is plenty.
    n_ctx: int = Field(default=4096, ge=512, le=131072)

    # CPU thread count. None ⇒ llama.cpp picks a reasonable default
    # based on logical cores. Override on shared hosts to leave room
    # for other apps.
    n_threads: int | None = None

    # Build-channel + commit, exposed via /healthz. Same convention
    # as the main vorrat app — keeps Watchtower / debug stories aligned.
    build_channel: str = "dev"
    build_sha: str = "unknown"
    build_date: str = "unknown"

    # Maximum tokens to generate per classification. The grammar
    # constrains us to one of 16 short identifiers; 16 covers them
    # all (the longest is `pasta_reis_koerner` at 18 chars ≈ 7 tokens).
    max_output_tokens: int = Field(default=24, ge=1, le=128)


def get_settings() -> Settings:
    """Module-level loader. FastAPI's lifespan instantiates once."""
    return Settings()
