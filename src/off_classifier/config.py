"""Runtime settings, loaded from environment variables.

Production layout: the lifespan auto-downloads the GGUF specified by
``model_repo``/``model_filename`` from HuggingFace Hub into the
``HF_HOME`` cache (defaults to ``/models/hf_cache`` in the container,
backed by a named volume). First start pulls ~4.4 GB; subsequent
starts hit the cache and load in seconds.

This matches the pattern used by sentence-transformers in vorrat
(``VORRAT_EMBEDDING_MODEL_ID=BAAI/bge-m3`` self-pulling into
``vorrat_canary_hf``). The optional ``model_path_override`` exists
for tests and ``docker cp``-style scenarios where the operator wants
to bypass the download entirely.
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

    # HuggingFace Hub source for auto-download. Empty string disables
    # the auto-download path entirely — the service then falls back
    # to model_path_override (if given) or the unloaded stub.
    # `bartowski` is the de-facto community-trusted GGUF repacker;
    # their build is imatrix-calibrated which improves Q4 quality
    # vs the upstream Qwen GGUF for the same nominal quantisation.
    model_repo: str = Field(default="bartowski/Qwen2.5-14B-Instruct-GGUF")

    # Specific quant + filename inside the repo. GGUF repos host
    # multiple variants (Q4_K_M, Q5_K_M, Q6_K, …); we have to name
    # the exact file. Q4_K_M is the recommended sweet-spot — for
    # 14B that lands at ~9 GB resident, near-Q5 quality. Bumped from
    # 7B (~4.4 GB) to 14B (~9 GB) on 2026-05-10 after a 25-item eval
    # of /lebensmittel showed 7B confabulating on German specialty
    # terms (Quark vs Yogurt, mildgesäuerte Butter, etc.) and
    # producing constructed en:-tags rather than using the OFF-tags
    # from the input verbatim. ADR-0038 §4.1 budgeted for this.
    model_filename: str = Field(default="Qwen2.5-14B-Instruct-Q4_K_M.gguf")

    # Optional explicit GGUF path, takes precedence over auto-download.
    # Useful for: (a) tests against a small model without HF, (b) air-
    # gapped hosts where the operator copied a GGUF in via docker cp,
    # (c) iterating on a custom finetune. Setting this disables the
    # auto-download entirely; the file is loaded if it exists, stub
    # otherwise.
    model_path_override: str | None = None

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
