# Changelog

## [Unreleased]

### Added

- Walking-skeleton: FastAPI app with `/healthz`, `/classify` (POST),
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
