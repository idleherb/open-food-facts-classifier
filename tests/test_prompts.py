"""Unit tests for the prompt + grammar builders."""

from __future__ import annotations

import pytest

from off_classifier.inference.prompts import (
    build_chat_messages,
    build_grammar,
    parse_response_text,
)
from off_classifier.schemas import ClassifyRequest
from off_classifier.taxonomy import all_categories


def test_build_grammar_lists_all_16_categories() -> None:
    grammar = build_grammar()
    for cat in all_categories():
        assert f'"{cat}"' in grammar
    assert grammar.startswith("root ::=")


def test_build_grammar_is_pipe_separated() -> None:
    """The grammar must list alternatives with `|`. If anyone refactors
    it into a sequence (`""`) by accident, this catches it.
    """
    grammar = build_grammar()
    pipe_count = grammar.count("|")
    assert pipe_count == len(all_categories()) - 1


def test_build_chat_messages_starts_with_system() -> None:
    msgs = build_chat_messages(ClassifyRequest(name="Mehl"))
    assert msgs[0]["role"] == "system"
    # System prompt must mention all 16 buckets so the model sees them.
    system = msgs[0]["content"]
    for cat in all_categories():
        assert cat in system


def test_build_chat_messages_includes_few_shot_pairs() -> None:
    msgs = build_chat_messages(ClassifyRequest(name="Mehl"))
    # Each example contributes (user, assistant); 15 examples ⇒ 30 turns
    # plus 1 system + 1 final user = 32 messages.
    assert len(msgs) == 32
    # Last message is the actual query.
    assert msgs[-1]["role"] == "user"
    assert "Mehl" in msgs[-1]["content"]


def test_build_chat_messages_omits_empty_optional_fields() -> None:
    """Empty fields must not leak into the prompt as 'Brand: None' etc."""
    msgs = build_chat_messages(ClassifyRequest(name="Mehl"))
    last = msgs[-1]["content"]
    assert "None" not in last
    assert "Marke:" not in last  # brand was omitted
    assert "Generisch:" not in last
    assert "Beschreibung:" not in last
    assert "OFF-Tags:" not in last


def test_build_chat_messages_renders_all_fields_when_provided() -> None:
    req = ClassifyRequest(
        name="Hafermilch",
        brand="Oatly",
        generic_name="Pflanzendrink",
        description="Haferdrink mit Säurregulator",
        off_categories_tags=["en:beverages", "en:plant-based-milks"],
    )
    msgs = build_chat_messages(req)
    last = msgs[-1]["content"]
    assert "Marke: Oatly" in last
    assert "Generisch: Pflanzendrink" in last
    assert "Beschreibung: Haferdrink" in last
    assert "OFF-Tags:" in last
    assert "en:plant-based-milks" in last


def test_build_chat_messages_truncates_off_tags_to_last_eight() -> None:
    """OFF returns up to 100+ tags; we keep only the most-specific tail."""
    tags = [f"en:tag-{i}" for i in range(20)]
    req = ClassifyRequest(name="Anything", off_categories_tags=tags)
    msgs = build_chat_messages(req)
    last = msgs[-1]["content"]
    # The earliest tags must be absent — they're the generic "en:foods"
    # noise and would crowd out the informative tail.
    assert "en:tag-0" not in last
    assert "en:tag-5" not in last
    # The last 8 must be present.
    for i in range(12, 20):
        assert f"en:tag-{i}" in last


def test_parse_response_text_accepts_valid_category() -> None:
    assert parse_response_text("mehl_backen") == "mehl_backen"


def test_parse_response_text_strips_whitespace() -> None:
    """Models occasionally leak trailing newlines despite the grammar."""
    assert parse_response_text("  milchprodukte\n") == "milchprodukte"


def test_parse_response_text_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="non-enum value"):
        parse_response_text("frischwaren")  # plausible-looking typo


def test_parse_response_text_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        parse_response_text("   ")
