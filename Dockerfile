# syntax=docker/dockerfile:1.7
# Multi-stage build for open-food-facts-classifier.
#
# llama-cpp-python builds CPU-only wheels by default — we install via
# uv with no extra build flags. The TrueNAS host runs amd64 only;
# arm64 is not a target. Model is mounted at runtime via Docker
# volume, never baked into the image (4.7 GB GGUF would balloon image
# size and force a re-push on every tag).

FROM python:3.12-slim AS builder

# llama-cpp-python ships sdist + a CPU-build path; the slim image has
# no compiler. Install build-essentials and cmake just for the builder
# stage; the runtime stage stays slim because we only copy /opt/venv
# and src/ over.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
    && rm -rf /var/lib/apt/lists/*

ARG UV_VERSION=0.11.4
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

WORKDIR /app
# README.md is referenced as `readme` in pyproject.toml — hatchling
# reads it during `uv pip install -e`. Without the COPY, the install
# bails with `OSError: Readme file does not exist: README.md`.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python -e ".[dev]"

COPY src ./src

FROM python:3.12-slim AS runtime

# Runtime deps for llama-cpp-python — needs libgomp for OpenMP-backed
# CPU threading. Skip build tools (already compiled in builder stage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/src ./src
COPY pyproject.toml ./

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"
# HuggingFace caches model snapshots under HF_HOME/hub/...
# Pointing it at /models means the GGUF lands in the named volume
# the operator mounts there, persisting across container restarts.
# First start downloads ~4.4 GB; subsequent starts hit the cache.
ENV HF_HOME="/models/hf_cache"
ENV PYTHONUNBUFFERED=1

# Pre-create /models/hf_cache with UID 1000 ownership inside the
# image. Compose stacks set `user: '1000:1000'`; without this the
# container's worker would hit `Permission denied` on the named
# volume because Docker would mount /models with root-owned
# defaults. When the operator's named volume is empty on first
# mount, Docker seeds it from the image — so these permissions
# carry over and HF can write the model cache. Existing volumes
# with broken perms must be wiped once before this fix takes effect.
RUN mkdir -p /models/hf_cache /app/data && \
    chown -R 1000:1000 /models /app

ARG VORRAT_BUILD_CHANNEL=dev
ARG VORRAT_BUILD_SHA=unknown
ARG VORRAT_BUILD_DATE=unknown
ENV OFF_CLASSIFIER_BUILD_CHANNEL=${VORRAT_BUILD_CHANNEL}
ENV OFF_CLASSIFIER_BUILD_SHA=${VORRAT_BUILD_SHA}
ENV OFF_CLASSIFIER_BUILD_DATE=${VORRAT_BUILD_DATE}

EXPOSE 8001

# uvicorn binds to 0.0.0.0 inside the container; the host's reverse
# proxy (Caddy with TLS) decides who gets to see it externally.
# --log-level warning silences the per-request access log (one INFO
# line per /healthz poll = 12/min from Watchtower + the vorrat app's
# heartbeat); per the parent vorrat CLAUDE.md hard rule 10 production
# log level is WARNING+errors only. Override with `--log-level info`
# at run time when actively debugging.
CMD ["uvicorn", "off_classifier.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "warning"]
