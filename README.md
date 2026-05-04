# open-food-facts-classifier

Local LLM-backed product-category classifier for [Vorrat](https://github.com/idleherb/vorrat).
Stage 2 of [Vorrat ADR-0031](https://github.com/idleherb/vorrat/blob/main/docs/architecture/adrs/0031-product-classifier-off-first-llm-fallback.md).

Runs Qwen 2.5-7B-Instruct (Q4_K_M GGUF, ~4.7 GB) via `llama-cpp-python`
on CPU and returns one of 16 stable category buckets, GBNF-grammar-constrained
so the model literally cannot return a value outside the enum.

This service is the fallback path when Vorrat's primary OFF-tag
classifier (in the main repo) cannot map a product. The two
classifiers stay in lock-step on the 16-bucket taxonomy via a unit
test that grep-checks the sibling vorrat repo when checked out.

## Architecture in 30 seconds

```
        Vorrat (main app)
           |
           | scan barcode → OFF lookup → categories_tags
           |   ↓
           | classify_via_off(tags)  ──┐
           |   |                      │
           |   +─ Some(category) ──────────────────→ persist
           |   |                      │
           |   +─ None ───────────────┘
           |                          ↓
           |        POST /classify    │
           +──────────────────────────┘
                       │
                       ↓
             open-food-facts-classifier
                       │
                       ↓
           Qwen 2.5-7B (GBNF-constrained)
                       ↓
                   one of 16 IDs
```

## Local development

```sh
# 1. install (uv handles the venv)
uv sync --extra dev

# 2. tests, no model required
uv run pytest

# 3. run the service in unloaded-stub mode (no GGUF needed)
uv run uvicorn off_classifier.main:app --reload --port 8001
curl localhost:8001/healthz   # ok=true, model_loaded=false
curl -X POST localhost:8001/classify \
  -H content-type:application/json \
  -d '{"name":"Mehl"}'
# 503 — model not loaded
```

## Running with the real model

```sh
mkdir -p models/
uv run huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
  --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models/

OFF_CLASSIFIER_MODEL_PATH="$(pwd)/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf" \
  uv run uvicorn off_classifier.main:app --port 8001
```

Model load takes ~10s on first start (mmap of 4.7 GB). After that,
each classification is sub-second on a modern x86_64 CPU
(~hundreds of ms).

## Production deployment

Single-track sidecar: every push to `main` builds + publishes `:latest`
to GHCR (`build.yml`). The consuming `vorrat-services` TrueNAS app
stack pulls `:latest`; Watchtower swaps the running container within
its poll interval. No canary/stable split, no version tags, no release
ceremony — sidecars are infrequently redeployed and the vorrat app
degrades gracefully if the classifier is briefly unreachable
(503 → "unklassifiziert" fall-through).

Compose snippet (lives in the `vorrat-services` stack on TrueNAS):

```yaml
services:
  classifier:
    image: ghcr.io/idleherb/open-food-facts-classifier:latest
    restart: unless-stopped
    volumes:
      - ./models:/models:ro
    environment:
      OFF_CLASSIFIER_MODEL_PATH: /models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
      OFF_CLASSIFIER_N_THREADS: "4"
    labels:
      com.centurylinklabs.watchtower.enable: "true"
    deploy:
      resources:
        limits:
          memory: 8G
```

## Configuration

Every setting is `OFF_CLASSIFIER_<UPPER_SNAKE>` in the environment.

| Setting              | Default                                              | Notes                                |
|----------------------|------------------------------------------------------|--------------------------------------|
| `MODEL_PATH`         | unset                                                | unset ⇒ /classify returns 503        |
| `N_CTX`              | 4096                                                 | Prompt is ~200 tokens, output 1      |
| `N_THREADS`          | unset (auto)                                         | Cap on shared hosts                  |
| `MAX_OUTPUT_TOKENS`  | 24                                                   | Longest enum is 7 tokens             |
| `BUILD_CHANNEL`      | `dev`                                                | Set by Dockerfile build-args         |
| `BUILD_SHA`          | `unknown`                                            | ditto                                |
| `BUILD_DATE`         | `unknown`                                            | ditto                                |

## Why this is a separate service

See ADR-0031 in vorrat for the full rationale. Short version:

- Different deployment lifecycle: model swaps rarely, app updates often.
- Different resource profile: 5 GB resident vs ~200 MB.
- Crash-isolation: a llama.cpp segfault must not take down the main app.
- Watchtower-driven auto-update of the main app would otherwise have
  to wait for a 4.7 GB image pull each time it ships.
