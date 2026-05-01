"""The 16-bucket UI taxonomy — kept byte-identical with the corresponding
file in the `vorrat` repo (`src/vorrat/classifier/categories.py`). When
either side bumps the taxonomy, both sides MUST bump in lock-step in
the same calendar day; the GBNF grammar in this service is generated
from this list, and a drift between services would let a category land
in the DB that the classifier can't return.

A future commit will likely extract the taxonomy into a tiny shared
package or a single OpenAPI-served endpoint; for now duplication keeps
each service deployable in isolation, which matters more during the
walking-skeleton phase.
"""

from __future__ import annotations

from typing import Literal

Category = Literal[
    "mehl_backen",
    "milchprodukte",
    "obst_gemuese",
    "konserven",
    "pasta_reis_koerner",
    "saucen_oele_gewuerze",
    "getraenke",
    "kaffee_tee",
    "suesswaren",
    "salzige_snacks",
    "fleisch_fisch",
    "tiefkuehl",
    "brot_backwaren",
    "babynahrung",
    "hygiene_haushalt",
    "sonstiges",
]


CATEGORY_DISPLAY_NAMES: dict[Category, str] = {
    "mehl_backen": "Mehl & Backen",
    "milchprodukte": "Milchprodukte",
    "obst_gemuese": "Obst & Gemüse",
    "konserven": "Konserven",
    "pasta_reis_koerner": "Pasta, Reis, Körner",
    "saucen_oele_gewuerze": "Saucen, Öle & Gewürze",
    "getraenke": "Getränke",
    "kaffee_tee": "Kaffee & Tee",
    "suesswaren": "Süßwaren",
    "salzige_snacks": "Salzige Snacks",
    "fleisch_fisch": "Fleisch & Fisch",
    "tiefkuehl": "Tiefkühl",
    "brot_backwaren": "Brot & Backwaren",
    "babynahrung": "Babynahrung",
    "hygiene_haushalt": "Hygiene & Haushalt",
    "sonstiges": "Sonstiges",
}


def all_categories() -> tuple[Category, ...]:
    return (
        "mehl_backen",
        "milchprodukte",
        "obst_gemuese",
        "konserven",
        "pasta_reis_koerner",
        "saucen_oele_gewuerze",
        "getraenke",
        "kaffee_tee",
        "suesswaren",
        "salzige_snacks",
        "fleisch_fisch",
        "tiefkuehl",
        "brot_backwaren",
        "babynahrung",
        "hygiene_haushalt",
        "sonstiges",
    )
