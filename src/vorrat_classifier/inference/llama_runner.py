"""llama-cpp-python implementation of ClassifierRunner.

This module is excluded from coverage in pyproject.toml — its
behaviour is verified by integration smoke tests with a real GGUF,
not by mocking llama_cpp internals. The protocol-level seam in
runner.py is what unit tests exercise.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from llama_cpp import Llama, LlamaGrammar

from vorrat_classifier.inference.prompts import (
    build_chat_messages,
    build_grammar,
    parse_response_text,
)
from vorrat_classifier.schemas import ClassifyRequest, ClassifyResponse


class LlamaCppRunner:
    """Loads a GGUF on construction; reuses it across requests.

    The Llama instance is mmap-backed, so one process holding it
    open costs roughly the in-memory footprint of the active layers
    (~5 GB for Qwen 2.5-7B Q4_K_M). Fork-safety is irrelevant — we
    run a single uvicorn worker; gunicorn-style preforking would
    require a different design.
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        max_output_tokens: int = 24,
    ) -> None:
        self._model_path = model_path
        self._max_output_tokens = max_output_tokens
        # chat_format=None ⇒ use the GGUF's embedded tokenizer.chat_template,
        # which is the correct ChatML-with-Qwen-tokens layout for
        # Qwen 2.5. Setting it explicitly to "qwen" would pick the old
        # Qwen-1 template — bug, not feature.
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format=None,
            verbose=False,
        )
        # Pre-build grammar once: it's a static enum, no per-request work.
        self._grammar = LlamaGrammar.from_string(build_grammar(), verbose=False)

    @property
    def model_id(self) -> str:
        # Use the file basename as the identifier — tracks the actual
        # GGUF the operator mounted, which may differ from what we
        # think it is (testing a custom finetune, etc.).
        return Path(self._model_path).stem

    @property
    def is_loaded(self) -> bool:
        return True

    def classify(self, req: ClassifyRequest) -> ClassifyResponse:
        messages = build_chat_messages(req)
        started = time.monotonic()
        # llama-cpp-python types `messages` as a union of TypedDicts
        # which dict[str, Any] doesn't structurally satisfy; cast at
        # the boundary keeps the prompt-builder free of llama-cpp
        # imports.
        result = self._llm.create_chat_completion(
            messages=cast("Any", messages),
            grammar=self._grammar,
            max_tokens=self._max_output_tokens,
            temperature=0.0,  # deterministic — same input ⇒ same output
            stream=False,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)
        # llama-cpp-python's chat-completion response shape mirrors the
        # OpenAI v1 schema: choices[0].message.content holds the text.
        # The return type is a Union[response, stream] — stream=False
        # narrows it at runtime but ty can't see that.
        response = cast("dict[str, Any]", result)
        text = response["choices"][0]["message"]["content"] or ""
        category = parse_response_text(text)
        return ClassifyResponse(
            category=category,
            model_id=self.model_id,
            inference_ms=elapsed_ms,
        )
