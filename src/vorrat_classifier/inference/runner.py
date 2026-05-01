"""Runner protocol — the boundary the FastAPI layer talks to.

Splitting the protocol from the llama-cpp-python implementation lets
tests swap a deterministic stub in via FastAPI dependency overrides,
which means the entire HTTP contract can be exercised without
downloading a 4.7 GB GGUF.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vorrat_classifier.schemas import ClassifyRequest, ClassifyResponse


@runtime_checkable
class ClassifierRunner(Protocol):
    """Anything that can turn a ClassifyRequest into a ClassifyResponse."""

    @property
    def model_id(self) -> str:
        """Stable identifier of the underlying model (e.g.
        ``Qwen2.5-7B-Instruct-Q4_K_M``). Echoed back in responses
        so callers can A/B across model bumps without re-deploying
        the wire schema.
        """

    @property
    def is_loaded(self) -> bool:
        """True iff the model is loaded and ``classify`` would succeed.
        False ⇒ HTTP layer returns 503; the deployment is alive but
        cannot serve its primary function (e.g. no GGUF mounted yet).
        """

    def classify(self, req: ClassifyRequest) -> ClassifyResponse:
        """Run classification synchronously.

        Implementations must populate ``inference_ms`` measuring only
        the time spent in the model call, not in prompt formatting
        or response parsing — that overhead is negligible and not
        what the caller wants visibility into.
        """
        ...
