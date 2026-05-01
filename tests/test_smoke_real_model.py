"""End-to-end smoke test against a real GGUF.

Skipped unless `OFF_CLASSIFIER_SMOKE_MODEL_PATH` is set in the env.
This is the only place the llama-cpp-python wrapper actually runs;
the rest of the suite uses a stub via the runner.py Protocol seam.

Usage:

    huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \\
        --include "Qwen2.5-7B-Instruct-Q4_K_M.gguf" \\
        --local-dir ./models/

    OFF_CLASSIFIER_SMOKE_MODEL_PATH=$(pwd)/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
        pytest tests/test_smoke_real_model.py -v --no-cov

Intentionally not part of the standard CI run — model load takes ~10s
and bumps the suite time from <1s to >15s, plus we don't want CI
runners pulling 4.7 GB on every push.
"""

from __future__ import annotations

import os

import pytest

from off_classifier.schemas import ClassifyRequest
from off_classifier.taxonomy import all_categories

MODEL_PATH = os.environ.get("OFF_CLASSIFIER_SMOKE_MODEL_PATH")

pytestmark = pytest.mark.skipif(
    not MODEL_PATH,
    reason="set OFF_CLASSIFIER_SMOKE_MODEL_PATH to run the real-model smoke test",
)


@pytest.fixture(scope="module")
def runner():  # type: ignore[no-untyped-def]
    """Module-scoped — loading a 7B model takes ~10s, do it once.

    Importing LlamaCppRunner inside the fixture (rather than at module
    top) keeps the rest of the suite from even touching llama_cpp
    when the smoke test is skipped.
    """
    from off_classifier.inference.llama_runner import (  # noqa: PLC0415
        LlamaCppRunner,
    )

    assert MODEL_PATH is not None  # narrowed by skipif
    return LlamaCppRunner(model_path=MODEL_PATH, n_ctx=4096)


def test_smoke_loads_and_returns_loaded(runner) -> None:  # type: ignore[no-untyped-def]
    assert runner.is_loaded is True
    assert runner.model_id  # non-empty


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Mehl Type 405", "mehl_backen"),
        ("Vollmilch 3.5%", "milchprodukte"),
        ("Bio-Bananen", "obst_gemuese"),
        ("Spaghetti", "pasta_reis_koerner"),
        ("Olivenöl", "saucen_oele_gewuerze"),
        ("Apfelsaft", "getraenke"),
        ("Espresso gemahlen", "kaffee_tee"),
        ("Schokoladentafel", "suesswaren"),
        ("Kartoffelchips", "salzige_snacks"),
        ("Hähnchenbrust", "fleisch_fisch"),
        ("TK-Pizza Salami", "tiefkuehl"),
        ("Vollkornbrot", "brot_backwaren"),
    ],
)
def test_smoke_classifies_obvious_cases(  # type: ignore[no-untyped-def]
    runner, name: str, expected: str
) -> None:
    """Hand-picked easy cases. The grammar guarantees the answer
    is *some* valid bucket; this test asserts the *right* one for
    cases where there's no ambiguity. If the model fails on a case
    here, the prompt + few-shots need rework — not the test.
    """
    response = runner.classify(ClassifyRequest(name=name))
    assert response.category in all_categories()
    assert response.category == expected, (
        f"{name!r}: got {response.category!r}, expected {expected!r}"
    )


def test_smoke_grammar_constraint_holds(runner) -> None:  # type: ignore[no-untyped-def]
    """Even with intentionally adversarial input, the GBNF grammar
    forces the response into the 16-bucket enum. If this ever fails
    the grammar is broken.
    """
    response = runner.classify(
        ClassifyRequest(name="???", brand="???", description="completely nonsensical input")
    )
    assert response.category in all_categories()
