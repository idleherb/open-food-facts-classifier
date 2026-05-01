# Changelog

## [Unreleased]

### Added

- **HuggingFace Hub auto-download for the GGUF.** New settings
  `OFF_CLASSIFIER_MODEL_REPO` (default `bartowski/Qwen2.5-7B-Instruct-GGUF`)
  + `OFF_CLASSIFIER_MODEL_FILENAME` (default `Qwen2.5-7B-Instruct-Q4_K_M.gguf`).
  At lifespan startup, `huggingface_hub.hf_hub_download` pulls the GGUF
  into the cache pointed at by `HF_HOME` (Dockerfile sets it to
  `/models/hf_cache`, backed by a named volume). First start is slow
  (~5 min, ~4.4 GB); subsequent starts hit cache. HF_TOKEN is read by
  `huggingface_hub` from the env automatically — same pattern as
  sentence-transformers in vorrat (`vorrat_canary_hf` volume + BAAI/bge-m3
  self-pulling).

  Set `OFF_CLASSIFIER_MODEL_REPO=""` to disable auto-download and use
  `OFF_CLASSIFIER_MODEL_PATH_OVERRIDE` instead (e.g. for `docker cp`
  workflows or air-gapped hosts). Setting neither lands on the stub.

  The `OFF_CLASSIFIER_MODEL_PATH` env var from the previous walking-
  skeleton release is gone — the path is computed from `HF_HOME` +
  the repo + filename.

- Fail-soft on HF errors. Network/DNS failures (`OSError`), HTTP errors
  from the Hub (404 / 401 / 403 / 5xx via `HfHubHTTPError`), or
  malformed repo IDs (`ValueError`) all log a warning and degrade to
  the unloaded stub instead of crashing the container at startup.
  Watchtower rolling forward to a known-bad release shouldn't take
  the service offline.

### Changed

- Tests run with `OFF_CLASSIFIER_MODEL_REPO=""` set in conftest.py;
  the auto-download path is mocked explicitly where verified. Suite
  still 0 network calls and 0 GGUF on disk.
- CI smoke (canary + stable) sets `OFF_CLASSIFIER_MODEL_REPO=""` on
  the smoke container so the runner doesn't pull 4.4 GB on every
  push. Real-model verification stays in the opt-in
  `tests/test_smoke_real_model.py`.

### Walking-skeleton baseline (initial release)

- FastAPI app with `/healthz`, `/classify` (POST),
  Pydantic schemas for the 16-bucket taxonomy, GBNF grammar generator,
  prompt builder with 15 few-shot examples, runner protocol, llama-cpp-python
  implementation. The service is deployable without a GGUF mounted —
  `/classify` returns 503 until `OFF_CLASSIFIER_MODEL_PATH` resolves
  to a file. Match Vorrat's healthz shape so logs line up across services.
- Multi-stage Dockerfile (uv builder → python:3.12-slim runtime, libgomp
  for OpenMP). Image runs CPU-only; arm64 not targeted (TrueNAS host
  is amd64).
- Test suite covers HTTP layer, prompt builder, grammar shape, taxonomy
  parity (asserts the 16 IDs match the sibling vorrat repo when checked
  out together). llama-cpp-python wrapper is excluded from coverage —
  exercised via integration smoke against a real GGUF.
