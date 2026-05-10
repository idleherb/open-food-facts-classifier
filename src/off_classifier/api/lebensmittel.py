"""POST /lebensmittel — Lebensmittel-classifier endpoint (vorrat ADR-0038).

Counterpart to /classify which returns one of the 15 stable UI buckets.
This endpoint returns a granular Lebensmittel-id namespaced as either
``en:<off-tag>`` or ``vorrat:<household-slug>``. The two endpoints
share a Llama runtime via the shared ClassifierRunner; the model and
GBNF grammars are pre-built once at lifespan startup and reused.

Resolution semantics mirror /classify:
  200 — model is loaded and produced an id.
  503 — model is not loaded (no GGUF mounted, or `model_path` not set).
  422 — request validation failed (handled by FastAPI / Pydantic).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from off_classifier.api.classify import RunnerDep
from off_classifier.schemas import LebensmittelRequest, LebensmittelResponse

router = APIRouter(tags=["lebensmittel"])


@router.post("/lebensmittel", response_model=LebensmittelResponse)
async def lebensmittel(payload: LebensmittelRequest, runner: RunnerDep) -> LebensmittelResponse:
    """Map a product to a Lebensmittel-id (vorrat ADR-0038 §2a barcode-meta path).

    Synchronous on the inside — Llama.create_chat_completion blocks
    the worker thread for ~hundreds of ms on CPU. Same pattern as
    /classify; QPS is low (a household scans new barcodes O(10) per
    day plus occasional batch-classify on a receipt-photo).
    """
    if not runner.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "classifier model not loaded — set OFF_CLASSIFIER_MODEL_PATH to a GGUF on disk"
            ),
        )
    return runner.lebensmittel(payload)
