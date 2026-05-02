"""Prompt + grammar building for the LLM classifier.

The whole point of using a grammar-constrained chat completion is
that the model literally cannot return anything outside the 16-bucket
enum — there is no "post-validate the response, fall back to
sonstiges if invalid" branch, because the alternative state is
unreachable. The grammar pins it down at the sampler level.

The prompt is intentionally short. Long instructions waste tokens
without helping a strongly instruction-tuned model like Qwen 2.5.
The few-shot examples are the lever that moves accuracy.
"""

from __future__ import annotations

from typing import Any

from off_classifier.schemas import ClassifyRequest
from off_classifier.taxonomy import (
    CATEGORY_DISPLAY_NAMES,
    Category,
    all_categories,
)

# A short rationale per bucket, surfaced in the system prompt so the
# model has a one-line definition of each class. These mirror the
# ADR-0031 §taxonomy section.
_BUCKET_DEFINITIONS: dict[Category, str] = {
    "mehl_backen": "Backzutaten: Mehl, Zucker, Hefe, Backpulver, Vanille, Streusel.",
    "milchprodukte": (
        "Milch, Joghurt, Quark, Sahne, Käse — auch Pflanzenmilch und vegane Käsealternativen."
    ),
    "obst_gemuese": "Frisches Obst und Gemüse, Salat, Kartoffeln, Pilze.",
    "konserven": (
        "Eingedoste Bohnen, Tomaten, Mais, Thunfisch, fertige Suppen in Dosen oder Gläsern."
    ),
    "pasta_reis_koerner": (
        "Trockene Nudeln, Reis, Couscous, Bulgur, Quinoa, Hülsenfrüchte trocken."
    ),
    "saucen_oele_gewuerze": ("Öl, Essig, Gewürze, Salz, Sojasauce, Ketchup, Senf, Würzpasten."),
    "getraenke": "Säfte, Limonade, Wasser, Bier, Wein, Smoothies (kein Kaffee, kein Tee).",
    "kaffee_tee": "Kaffeebohnen, Kaffeepulver, Espresso-Pads, lose Tees, Teebeutel.",
    "suesswaren": ("Schokolade, Bonbons, Kekse, Pralinen, Müsliriegel mit Süßungsschwerpunkt."),
    "salzige_snacks": "Chips, Nüsse gesalzen, Cracker, Popcorn salzig, Brezeln.",
    "fleisch_fisch": (
        "Frisches oder verpacktes Fleisch, Wurst, Schinken, frischer oder geräucherter Fisch."
    ),
    "tiefkuehl": "Tiefkühlware: TK-Pizza, TK-Gemüse, Eis, TK-Fertiggerichte.",
    "brot_backwaren": "Brot, Brötchen, Toast, Croissants, fertige Backwaren.",
    "hygiene_haushalt": (
        "Waschmittel, Reiniger, Toilettenpapier, Zahnpasta, Seife — kein Lebensmittel."
    ),
    "sonstiges": ("Wirklich nichts davon passt — sehr selten gewählt, nur als letzter Ausweg."),
}


# Few-shot examples: real-shaped product descriptions paired with the
# correct bucket. The model sees these BEFORE the new product, so it
# anchors on the format. Examples are chosen to span easy + tricky
# cases (plant milk → milchprodukte landing in the dairy bucket as a
# substitute, TK-anything overriding the inner category).
_FEW_SHOT_EXAMPLES: tuple[tuple[ClassifyRequest, Category], ...] = (
    (
        ClassifyRequest(name="Mehl Type 405", brand="Aurora"),
        "mehl_backen",
    ),
    (
        ClassifyRequest(name="Hafermilch Barista", brand="Oatly", generic_name="Pflanzendrink"),
        "milchprodukte",
    ),
    (
        ClassifyRequest(name="Bio-Bananen", brand=None, generic_name="Frische Früchte"),
        "obst_gemuese",
    ),
    (
        ClassifyRequest(
            name="Kichererbsen 400g", brand="dennree", generic_name="Hülsenfrüchte in Dose"
        ),
        "konserven",
    ),
    (
        ClassifyRequest(name="Spaghetti No.5", brand="Barilla"),
        "pasta_reis_koerner",
    ),
    (
        ClassifyRequest(name="Olivenöl extra vergine", brand="Bertolli"),
        "saucen_oele_gewuerze",
    ),
    (
        ClassifyRequest(name="Apfelsaft naturtrüb 1L", brand="Voelkel"),
        "getraenke",
    ),
    (
        ClassifyRequest(name="Espresso 250g gemahlen", brand="Lavazza"),
        "kaffee_tee",
    ),
    (
        ClassifyRequest(name="Zartbitterschokolade 70%", brand="Lindt"),
        "suesswaren",
    ),
    (
        ClassifyRequest(name="Kartoffelchips Paprika", brand="funny-frisch"),
        "salzige_snacks",
    ),
    (
        ClassifyRequest(name="Bio-Hähnchenbrust", brand=None, generic_name="Geflügelfleisch"),
        "fleisch_fisch",
    ),
    (
        ClassifyRequest(name="TK-Pizza Salami", brand="Dr. Oetker"),
        "tiefkuehl",
    ),
    (
        ClassifyRequest(name="Vollkornbrot", brand="Mestemacher"),
        "brot_backwaren",
    ),
    (
        ClassifyRequest(name="Spülmaschinentabs", brand="Finish"),
        "hygiene_haushalt",
    ),
)


def _format_product(req: ClassifyRequest) -> str:
    """Render a request as the labelled block the prompt expects.

    Empty fields are dropped entirely — feeding the model "Brand:
    None" is worse than just not mentioning brand at all (it leaks
    English Python semantics into a German-language reasoning chain).
    """
    parts: list[str] = [f"Name: {req.name}"]
    if req.brand:
        parts.append(f"Marke: {req.brand}")
    if req.generic_name:
        parts.append(f"Generisch: {req.generic_name}")
    if req.description:
        # Truncate any extreme description (we already enforced 500
        # chars at the schema level; this is defence-in-depth in
        # case the schema is bypassed in tests).
        snippet = req.description[:500]
        parts.append(f"Beschreibung: {snippet}")
    if req.off_categories_tags:
        # Last few tags only — OFF orders general → specific, so
        # the tail carries the most informative signal.
        tail = ", ".join(req.off_categories_tags[-8:])
        parts.append(f"OFF-Tags: {tail}")
    return "\n".join(parts)


def _system_message() -> str:
    bucket_lines = "\n".join(
        f"- {cat}: {CATEGORY_DISPLAY_NAMES[cat]} — {_BUCKET_DEFINITIONS[cat]}"
        for cat in all_categories()
    )
    return (
        "Du klassifizierst Lebensmittel und Haushaltsprodukte in genau eine "
        "von 15 Kategorien für eine Vorratsapp. Antworte NUR mit der ID der "
        "Kategorie (snake_case, ohne Anführungszeichen, ohne weitere Wörter).\n\n"
        "Kategorien:\n"
        f"{bucket_lines}\n\n"
        "Wichtige Regeln:\n"
        "- Pflanzenmilch und vegane Käsealternativen → milchprodukte (Substitute "
        "  landen im Bucket der Original-Kategorie).\n"
        "- Tiefkühlware schlägt die Inhaltskategorie (TK-Lasagne ist tiefkuehl, "
        "  nicht fleisch_fisch).\n"
        "- Verwende sonstiges nur, wenn wirklich nichts passt (sehr selten)."
    )


def build_chat_messages(req: ClassifyRequest) -> list[dict[str, Any]]:
    """Assemble the full chat-completion message sequence.

    Format: one system turn with rules, then alternating user/assistant
    turns for each few-shot example, then the actual user query. This
    is the standard Qwen-2.5 chat layout and matches the embedded
    chat_template inside the GGUF — we don't pre-format the prompt.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_message()},
    ]
    for example_req, example_label in _FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": _format_product(example_req)})
        messages.append({"role": "assistant", "content": example_label})
    messages.append({"role": "user", "content": _format_product(req)})
    return messages


def build_grammar() -> str:
    """Generate a GBNF grammar restricting output to the 16 enum IDs.

    The grammar is a single rule listing the alternatives. Because
    llama.cpp applies the grammar at the sampler level (token-by-
    token), the model literally cannot emit an id outside this set —
    no post-validation needed.
    """
    alternatives = " | ".join(f'"{cat}"' for cat in all_categories())
    return f"root ::= ({alternatives})\n"


def parse_response_text(text: str) -> Category:
    """Strip whitespace + validate. Despite the grammar constraint,
    we still defensively check — if we ever swap to a different
    backend (e.g. ollama, which uses JSON Schema instead of GBNF),
    this is the single place that decides "did the model behave?".
    """
    clean = text.strip()
    for cat in all_categories():
        # Iterate so the type checker can narrow `cat` to the Literal
        # union; equality with `clean` is enough to pin it down.
        if clean == cat:
            return cat
    raise ValueError(f"model returned non-enum value {clean!r}")
