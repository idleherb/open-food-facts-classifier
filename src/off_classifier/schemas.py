"""Wire-level request/response shapes for the classify endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from off_classifier.taxonomy import Category


class ClassifyRequest(BaseModel):
    """Inputs to a classification call.

    We accept a structured payload rather than a single concatenated
    string so the prompt template can render each field with the
    appropriate label — this measurably improves accuracy on small
    LLMs vs feeding them a CSV blob.

    `name` is the only required field. Everything else is optional;
    OFF often returns sparse rows and the prompt template degrades
    gracefully (omits the section entirely if the field is absent).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=200)
    generic_name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=500)
    # OFF categories_tags — even though OFF-tag classification has
    # already failed by the time we get here (otherwise the LLM
    # wouldn't be called), the tags themselves are still informative
    # context for the LLM. They constrain the search space.
    off_categories_tags: list[str] | None = Field(default=None, max_length=64)


class ClassifyResponse(BaseModel):
    """Result of a single classification call.

    `category` is one of the 16 stable IDs — the GBNF grammar makes
    this a structural guarantee, not a runtime hope. `model_id` lets
    the caller log which model produced the answer; useful for A/B
    comparisons across model bumps.
    """

    category: Category
    model_id: str
    # Time spent inside `Llama.create_chat_completion` for this call,
    # excluding HTTP overhead. Surfaced so the caller (vorrat) can
    # decide on per-request retries, queue priorities, etc. without
    # having to time the call itself.
    inference_ms: int


class HealthzResponse(BaseModel):
    """Mirror of vorrat's /healthz shape — one less translation in logs."""

    ok: bool
    channel: str
    version: str
    commit: str
    # Whether the model is actually loaded. False ⇒ /classify will
    # return 503; the deployment is alive but unable to serve its
    # primary function. Useful for orchestration ("is this pod ready?").
    model_loaded: bool


class LebensmittelRequest(BaseModel):
    """Inputs to a Lebensmittel-classification call (vorrat ADR-0038 §2a).

    Same shape as ``ClassifyRequest`` for the barcode-meta input path
    so consumers can build on existing OFF integration. The fuller
    receipt-token batch shape from ADR-0038 §2b is a separate slice;
    this initial iteration ships the barcode-meta-shape only and is
    enough to characterise classifier accuracy on the Lebensmittel
    abstraction before we expand the consumer side.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    brand: str | None = Field(default=None, max_length=200)
    generic_name: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=500)
    off_categories_tags: list[str] | None = Field(default=None, max_length=64)


class LebensmittelResponse(BaseModel):
    """Result of a single Lebensmittel-classification call.

    `lebensmittel_id` is namespaced: `en:<off-tag-name>` for entries
    that map to an Open Food Facts taxonomy node (the bulk of common
    pantry items), `vorrat:<household-slug>` for niche or private-label
    items the OFF taxonomy doesn't cover. The grammar guarantees one
    of those two namespaces and a well-formed slug, but the slug
    itself is not enumerated — the model may produce a previously-
    unseen tag, which the consumer treats as a household-specific
    new entry. Display-label resolution (German short-string) is the
    consumer's responsibility (look up OFF taxonomy `de:` translation
    or accept the user's free-text label).

    The fuller ADR-0038 §2a response shape (with `alternatives` and a
    `source ∈ {off, llm, correction}` discriminator) is a follow-up
    slice; this initial endpoint surfaces only the proposal.
    """

    lebensmittel_id: str
    model_id: str
    inference_ms: int
