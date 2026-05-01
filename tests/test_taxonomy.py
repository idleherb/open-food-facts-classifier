"""Lock the 16-bucket taxonomy in lock-step with vorrat's copy."""

from __future__ import annotations

from pathlib import Path

from vorrat_classifier.taxonomy import (
    CATEGORY_DISPLAY_NAMES,
    all_categories,
)


def test_taxonomy_has_exactly_16_buckets() -> None:
    """ADR-0031 fixes the count at 16. A change here without a
    matching ADR + drop-in change in vorrat is a bug.
    """
    assert len(all_categories()) == 16


def test_taxonomy_ids_are_unique() -> None:
    cats = all_categories()
    assert len(set(cats)) == len(cats)


def test_display_names_cover_every_id() -> None:
    for cat in all_categories():
        assert cat in CATEGORY_DISPLAY_NAMES
        assert CATEGORY_DISPLAY_NAMES[cat]


def test_display_names_have_no_extra_keys() -> None:
    """Catches the symmetric drift: a name added to display dict for
    a bucket no longer in `all_categories`.
    """
    extras = set(CATEGORY_DISPLAY_NAMES.keys()) - set(all_categories())
    assert not extras


def test_taxonomy_matches_vorrat_repo() -> None:
    """The two services MUST stay byte-aligned on the bucket list,
    since the vorrat DB stores these IDs and this service generates
    a GBNF grammar from them. A drift here would let a category land
    in the DB that the LLM cannot return — which would force a
    schema-level migration to recover.

    Verified by reading vorrat's `categories.py` if it's a sibling
    on disk. Skipped when the sibling isn't there (e.g. CI checkout
    of just the classifier repo).
    """
    vorrat_categories = (
        Path(__file__).resolve().parents[2]
        / "vorrat"
        / "src"
        / "vorrat"
        / "classifier"
        / "categories.py"
    )
    if not vorrat_categories.exists():
        return  # not a co-located checkout; skip silently
    text = vorrat_categories.read_text()
    for cat in all_categories():
        assert f'"{cat}"' in text, f"vorrat repo missing {cat!r}"
