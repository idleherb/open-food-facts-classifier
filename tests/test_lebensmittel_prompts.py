"""Tests for the Lebensmittel prompt + grammar + parser."""

from __future__ import annotations

import pytest

from off_classifier.inference.lebensmittel_prompts import (
    build_lebensmittel_chat_messages,
    build_lebensmittel_grammar,
    parse_lebensmittel_response,
)
from off_classifier.schemas import LebensmittelRequest


def test_build_chat_messages_includes_system_and_few_shots() -> None:
    req = LebensmittelRequest(name="Mehl Type 405", brand="Aurora")
    messages = build_lebensmittel_chat_messages(req)
    assert messages[0]["role"] == "system"
    # System prompt mentions the two namespaces.
    assert "en:" in messages[0]["content"]
    assert "vorrat:" in messages[0]["content"]
    # Few-shots: alternating user/assistant pairs.
    assert len(messages) > 5
    # Last message is the actual request.
    assert messages[-1]["role"] == "user"
    assert "Mehl Type 405" in messages[-1]["content"]
    assert "Aurora" in messages[-1]["content"]


def test_build_chat_messages_drops_empty_optional_fields() -> None:
    req = LebensmittelRequest(name="Bananen")
    messages = build_lebensmittel_chat_messages(req)
    last_user = messages[-1]["content"]
    assert "Marke:" not in last_user
    assert "Generisch:" not in last_user
    assert "OFF-Tags:" not in last_user


def test_build_chat_messages_emits_off_tags_tail_only() -> None:
    """OFF tags are general → specific; the prompt drops the broad
    head and keeps only the last few tags."""
    long_tags = [f"en:tag{i}" for i in range(15)]
    req = LebensmittelRequest(name="X", off_categories_tags=long_tags)
    messages = build_lebensmittel_chat_messages(req)
    last_user = messages[-1]["content"]
    # Tail of 8 makes it in.
    assert "en:tag14" in last_user
    assert "en:tag7" in last_user
    # Earlier tags don't.
    assert "en:tag0" not in last_user


def test_grammar_is_well_formed_gbnf() -> None:
    grammar = build_lebensmittel_grammar()
    # Grammar must define a 'root' rule.
    assert "root" in grammar
    # Both namespaces must be alternatives in the prefix rule.
    assert '"en:"' in grammar
    assert '"vorrat:"' in grammar


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("en:fusilli", "en:fusilli"),
        ("en:whole-milk", "en:whole-milk"),
        ("vorrat:schlemmerfilet-cordon-bleu", "vorrat:schlemmerfilet-cordon-bleu"),
        # Whitespace around ⇒ stripped.
        ("  en:cheddar  \n", "en:cheddar"),
    ],
)
def test_parse_response_accepts_valid_ids(text: str, expected: str) -> None:
    assert parse_lebensmittel_response(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "fusilli",  # no namespace prefix
        "https:fusilli",  # wrong namespace
        "en:Fusilli",  # uppercase in slug
        "en:f",  # too short slug
        "en:" + "a" * 41,  # too long slug
        "en:fusilli with extra words",  # extra content after slug
        "vorrat:has_underscores",  # underscore not allowed
    ],
)
def test_parse_response_rejects_invalid_ids(text: str) -> None:
    with pytest.raises(ValueError):
        parse_lebensmittel_response(text)
