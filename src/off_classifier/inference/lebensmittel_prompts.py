"""Prompt + grammar building for the Lebensmittel-classifier path.

Counterpart to `prompts.py`, which builds the 15-bucket category prompt
and grammar. This module targets vorrat ADR-0038's `/lebensmittel`
output: one of two namespaces (`en:<off-tag>` or `vorrat:<slug>`) plus
a well-formed slug. The grammar is permissive on slug content (any
lowercase + dashes, length 3-40) — we don't enumerate the OFF
taxonomy here, the model is trusted to produce a reasonable tag
either matching OFF or coining a slug.

Few-shot examples are tuned for realistic German pantry products so
the model learns the granularity rule from ADR-0037 §1: form +
ingredient base distinguishes Lebensmittel; brand and Bio-status do
not.
"""

from __future__ import annotations

from typing import Any

from off_classifier.schemas import LebensmittelRequest

# Few-shot examples: (request, expected lebensmittel-id). Drawn from
# Eric's 2026-05-09 Kaufland receipt plus typical pantry items, with
# OFF-style en: tags where the OFF taxonomy has a clean leaf, and
# more granular vorrat: slugs where it doesn't or where Eric's
# household needs finer detail.
_FEW_SHOT_EXAMPLES: tuple[tuple[LebensmittelRequest, str], ...] = (
    (
        LebensmittelRequest(name="Mehl Type 405", brand="Aurora"),
        "en:wheat-flour-type-405",
    ),
    (
        LebensmittelRequest(name="Bio-Vollmilch", brand="Berchtesgadener Land"),
        "en:whole-milk",
    ),
    (
        LebensmittelRequest(name="Hafermilch Barista", brand="Oatly", generic_name="Pflanzendrink"),
        "en:oat-milk",
    ),
    (
        LebensmittelRequest(
            name="Spaghetti No.5",
            brand="Barilla",
            off_categories_tags=["en:cereals", "en:pastas", "en:wheat-pasta", "en:spaghetti"],
        ),
        "en:wheat-spaghetti",
    ),
    (
        LebensmittelRequest(
            name="Bio Fusilli Vollkorn",
            brand="Alnatura",
            off_categories_tags=["en:pastas", "en:wholegrain-pastas", "en:fusilli"],
        ),
        "en:wholemeal-fusilli",
    ),
    (
        LebensmittelRequest(
            name="Bio Farfalle",
            brand="De Cecco",
            off_categories_tags=["en:pastas", "en:wheat-pasta", "en:farfalle"],
        ),
        "en:wheat-farfalle",
    ),
    (
        LebensmittelRequest(name="Bio-Bananen", brand=None, generic_name="Frische Früchte"),
        "en:bananas",
    ),
    (
        LebensmittelRequest(name="Salzige Butter", brand="Kerrygold"),
        "en:salted-butter",
    ),
    (
        LebensmittelRequest(name="Cheddar", brand="Kerrygold"),
        "en:cheddar",
    ),
    (
        LebensmittelRequest(
            name="Tofu Natur",
            brand="Taifun",
            off_categories_tags=["en:plant-based-foods", "en:tofu", "en:firm-tofu"],
        ),
        "en:firm-tofu",
    ),
    (
        LebensmittelRequest(name="Wildlachs geräuchert", brand="Friedrichs"),
        "en:smoked-salmon",
    ),
    (
        LebensmittelRequest(name="Mittelscharfer Senf", brand="Löwensenf"),
        "en:mustard-medium",
    ),
    (
        LebensmittelRequest(name="Espresso 250g gemahlen", brand="Lavazza"),
        "en:ground-coffee",
    ),
    (
        LebensmittelRequest(
            name="Frosta Schlemmerfilet Cordon Bleu",
            brand="Frosta",
            off_categories_tags=["en:frozen-fish", "en:breaded-fish-fillets"],
        ),
        "vorrat:schlemmerfilet-cordon-bleu",
    ),
    (
        LebensmittelRequest(name="Kartoffelchips Paprika", brand="funny-frisch"),
        "en:paprika-crisps",
    ),
    # Negative-pattern shots: explicit demonstrations of the rules
    # that the v1 prompt got wrong on the 2026-05-10 eval. Each pattern
    # is shown via a product that is NOT in the eval-set, so the next
    # eval pass measures generalisation rather than memorisation.
    (
        # Bio-strip even when "bio" is the leading word in name + brand.
        # (Pattern A: bio leaked into the slug in 2/25 eval cases.)
        LebensmittelRequest(name="Bio Salami Mailaender Art", brand="Edeka Bio"),
        "en:salami",
    ),
    (
        # Brand-strip in the vorrat: namespace too. (Pattern B: brand
        # leaked into a vorrat: slug in 1/25 eval cases.)
        LebensmittelRequest(name="Iglo Bratkartoffeln Tiefkuehl", brand="Iglo"),
        "vorrat:bratkartoffeln",
    ),
    (
        # Packaging/cut info ("halbiert") does NOT enter the ID.
        # (Pattern C: cut info leaked in 1/25.)
        LebensmittelRequest(name="Brokkoli halbiert", generic_name="Frischer Brokkoli"),
        "en:broccoli",
    ),
    (
        # OFF tag explicitly present ⇒ MUST be used as en:; never
        # invent a vorrat: slug when an OFF tag fits. (Pattern G:
        # available OFF tag bypassed in 2/25.)
        LebensmittelRequest(
            name="Forelle gerauchert",
            brand="Deutsche See",
            off_categories_tags=["en:fish", "en:smoked-fish", "en:smoked-trout"],
        ),
        "en:smoked-trout",
    ),
    (
        # Non-food: NEVER hallucinate an en:-tag for a concept outside
        # the OFF taxonomy. The vorrat: namespace is the right place.
        # (Pattern E: en:-tag hallucinated for non-food in 1/25.)
        LebensmittelRequest(
            name="Spuelmittel Zitrone",
            brand="Pril",
            off_categories_tags=None,
        ),
        "vorrat:dish-soap",
    ),
)


def _format_product(req: LebensmittelRequest) -> str:
    """Render the request as the labelled block the prompt expects."""
    parts: list[str] = [f"Name: {req.name}"]
    if req.brand:
        parts.append(f"Marke: {req.brand}")
    if req.generic_name:
        parts.append(f"Generisch: {req.generic_name}")
    if req.description:
        snippet = req.description[:500]
        parts.append(f"Beschreibung: {snippet}")
    if req.off_categories_tags:
        tail = ", ".join(req.off_categories_tags[-8:])
        parts.append(f"OFF-Tags: {tail}")
    return "\n".join(parts)


def _system_message() -> str:
    return (
        "Du klassifizierst Produkte zu einer Lebensmittel-Abstraktion fuer "
        "eine Vorratsapp. Antworte NUR mit der ID des Lebensmittels.\n\n"
        "Format der ID:\n"
        "- en:<off-tag-name>  fuer Produkte mit Eintrag in der Open-Food-Facts-"
        "  Taxonomie (z.B. en:fusilli, en:whole-milk, en:cheddar).\n"
        "- vorrat:<slug>      fuer Produkte OHNE OFF-Entsprechung "
        "  (Marken-Fertiggerichte, Spezialitaeten) UND fuer Produkte die kein "
        "  Lebensmittel sind (Blumen, Drogerie, Zeitschriften).\n\n"
        "Slug-Regeln: lowercase, ASCII, Bindestriche, Laenge 3-40.\n\n"
        "Granularitaets-Regeln (HART, niemals brechen):\n"
        "- Form + Zutatenbasis bestimmen das Lebensmittel: Fusilli != Penne, "
        "Vollkorn != Weizen.\n"
        "- Marke NIEMALS in der ID — auch nicht im vorrat:-Slug.\n"
        "  Beispiel: Iglo Spinat → en:spinach (NICHT vorrat:iglo-spinat).\n"
        "- Bio NIEMALS in der ID.\n"
        "  Beispiel: Bio-Salami → en:salami (NICHT en:bio-salami).\n"
        "  Beispiel: Bio-Karotten → en:carrots (NICHT en:bio-carrots).\n"
        "- Verpackungsgroesse, Verpackungsform und Zuschnitt (Stueck, Scheiben, "
        "halbiert, ganz, gerieben, fein, grob, dick, duenn) gehoeren NICHT in die ID.\n"
        "  Beispiel: Brokkoli halbiert → en:broccoli (NICHT en:broccoli-halved).\n"
        "- Wenn off_categories_tags einen passenden Tag enthaelt, MUSS dieser als "
        "en:-ID verwendet werden — niemals einen vorrat:-Slug erfinden, wenn ein "
        "OFF-Tag passt.\n"
        "- Wenn das Produkt KEIN Lebensmittel ist (Schnittblumen, Putzmittel, "
        "Zeitschrift), nutze vorrat:<slug> — niemals einen erfundenen en:-Tag fuer "
        "Konzepte ausserhalb der OFF-Taxonomie.\n"
        "- Bei OFF-Tags: bevorzuge den spezifischsten passenden Tag, oft am Ende "
        "der Liste.\n"
        "- Wenn unsicher zwischen spezifisch und generisch: lieber generischer "
        "(en:potatoes statt en:waxy-bio-potatoes-from-uckermark)."
    )


def build_lebensmittel_chat_messages(req: LebensmittelRequest) -> list[dict[str, Any]]:
    """Assemble the chat-completion messages for a Lebensmittel call.

    Same shape as the category prompt: system turn → alternating
    user/assistant few-shot turns → the actual user query.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_message()},
    ]
    for example_req, example_id in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": _format_product(example_req)})
        messages.append({"role": "assistant", "content": example_id})
    messages.append({"role": "user", "content": _format_product(req)})
    return messages


def build_lebensmittel_grammar() -> str:
    """GBNF grammar restricting output to a well-formed Lebensmittel ID.

    Output shape: ``(en|vorrat):<slug>`` where slug is lowercase ASCII
    + dashes, length 3..40. The grammar pins this at the sampler level
    so the model literally cannot emit malformed IDs (extra whitespace,
    uppercase, invalid characters, missing prefix). The slug content
    itself is permissive — we don't enumerate the OFF taxonomy here,
    the model is trusted to produce reasonable tags from prompt context.
    """
    return 'root ::= prefix slug\nprefix ::= "en:" | "vorrat:"\nslug ::= [a-z] [a-z0-9-]{2,39}\n'


# Slug length bounds — must match the GBNF grammar in build_lebensmittel_grammar().
# The grammar emits `[a-z] [a-z0-9-]{2,39}` ⇒ overall slug length 3..40.
_SLUG_MIN_LEN = 3
_SLUG_MAX_LEN = 40


def parse_lebensmittel_response(text: str) -> str:
    """Strip whitespace and validate the response is shape-correct.

    Defensive even though the GBNF guarantees this — if a future
    backend swap drops grammar support, this is the single place
    that decides 'did the model behave?'.
    """
    clean = text.strip()
    if not (clean.startswith("en:") or clean.startswith("vorrat:")):
        raise ValueError(f"model returned non-namespaced lebensmittel-id {clean!r}")
    prefix, _, slug = clean.partition(":")
    if not slug or not (_SLUG_MIN_LEN <= len(slug) <= _SLUG_MAX_LEN):
        raise ValueError(f"model returned malformed slug in {clean!r}")
    if not slug.replace("-", "").isalnum() or not slug[0].isalpha() or not slug[0].islower():
        raise ValueError(f"model returned malformed slug {slug!r} in {clean!r}")
    return f"{prefix}:{slug}"
