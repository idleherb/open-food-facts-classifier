"""POST /classify — the only feature endpoint of this service.

Resolution semantics:
  200 — model is loaded and produced a category.
  503 — model is not loaded (no GGUF mounted, or `model_path` not set).
  422 — request validation failed (handled by FastAPI / Pydantic).

We deliberately do NOT return 5xx for "model produced a non-enum
value" — the GBNF grammar makes that unreachable, and if it ever
does happen we'd rather crash loudly via the runner's ValueError
than swallow it as a 500.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from off_classifier.inference.runner import ClassifierRunner
from off_classifier.schemas import ClassifyRequest, ClassifyResponse

router = APIRouter(tags=["classify"])


def get_runner(request: Request) -> ClassifierRunner:
    """Resolve the ClassifierRunner hung off ``app.state`` by lifespan.

    Tests override this dependency to inject a deterministic stub
    runner that doesn't need a GGUF on disk.
    """
    runner: ClassifierRunner | None = getattr(request.app.state, "runner", None)
    if runner is None:
        # Lifespan didn't register a runner at all (e.g. import-time
        # crash). Surface as 503 — there is no functional service to
        # talk to, but health endpoints can still answer.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="classifier runner not initialised",
        )
    return runner


RunnerDep = Annotated[ClassifierRunner, Depends(get_runner)]


@router.post("/classify", response_model=ClassifyResponse)
async def classify(payload: ClassifyRequest, runner: RunnerDep) -> ClassifyResponse:
    """Classify a single product into one of the 16 stable buckets.

    Synchronous on the inside — Llama.create_chat_completion blocks
    the worker thread for ~hundreds of ms on CPU. Acceptable for the
    expected QPS (a household scans new barcodes O(10) per day);
    if QPS ever materially rises we'd switch to a job-queue pattern,
    not move llama-cpp to a thread pool (still single-process).
    """
    if not runner.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "classifier model not loaded — set OFF_CLASSIFIER_MODEL_PATH to a GGUF on disk"
            ),
        )
    return runner.classify(payload)
