"""Synthetic draft instrument generator with ground-truth defect labels,
used to evaluate instrument_gate.py's precision and recall (the gate has
no other source of "truth" to be checked against; a synthetic generator
with known labels is the reference oracle for this module, the same role
an independently-coded reference implementation plays elsewhere in this
portfolio).
"""
from __future__ import annotations

import random

from sp.instrument_gate import DraftInstrument, SurveyItem

CLEAN_ITEM_TEMPLATES = [
    "How satisfied are you with the relevance of your home feed?",
    "How often do you save a Pin after seeing it in your feed?",
    "How easy or difficult is it to find boards related to your interests?",
    "How would you rate the visual quality of the Pins you see?",
    "How likely are you to recommend this app to a friend?",
    "How relevant do you find the search results for your queries?",
]

DEFECT_BUILDERS = {
    "double_barreled": lambda: SurveyItem(
        item_id="x",
        text="How satisfied are you with the app's speed and its design?",
        item_type="likert_balanced",
    ),
    "leading_wording": lambda: SurveyItem(
        item_id="x",
        text="Don't you agree that the new feed design is a clear improvement?",
        item_type="likert_agree_disagree",
    ),
    "absolute_wording": lambda: SurveyItem(
        item_id="x",
        text="The recommendations are always relevant to my interests.",
        item_type="likert_agree_disagree",
    ),
    "missing_dont_know_option": lambda: SurveyItem(
        item_id="x",
        text="How useful is the new collaborative board feature?",
        item_type="likert_balanced",
        has_dont_know_option=False,
    ),
    "overlong_grid": lambda: SurveyItem(
        item_id="x",
        text="Rate each of the following 20 features.",
        item_type="grid",
        grid_row_count=20,
    ),
    "inconsistent_scale_direction": "SPECIAL",  # handled at instrument level
}

DEFECT_NAMES = list(DEFECT_BUILDERS.keys())


def _clean_item(idx: int, rng: random.Random) -> SurveyItem:
    text = rng.choice(CLEAN_ITEM_TEMPLATES)
    return SurveyItem(item_id=f"q{idx}", text=text, item_type="likert_balanced")


def _build_instrument(
    instrument_id: str, defect_names: list[str], rng: random.Random
) -> DraftInstrument:
    n_clean_items = rng.randint(3, 5)
    items = [_clean_item(i, rng) for i in range(n_clean_items)]

    for defect_name in defect_names:
        if defect_name == "inconsistent_scale_direction":
            items.append(
                SurveyItem(
                    item_id="q_dir_a",
                    text="How positive is your experience with the new layout?",
                    item_type="likert_balanced",
                    scale_direction="positive_first",
                )
            )
            items.append(
                SurveyItem(
                    item_id="q_dir_b",
                    text="How negative is your experience with search results?",
                    item_type="likert_balanced",
                    scale_direction="negative_first",
                )
            )
        else:
            item = DEFECT_BUILDERS[defect_name]()
            item.item_id = f"defect_{defect_name}"
            items.append(item)

    for i, item in enumerate(items):
        if item.item_id == "x":
            item.item_id = f"q{i}"

    return DraftInstrument(
        instrument_id=instrument_id, items=items, seeded_defects=list(defect_names)
    )


def generate_draft_instruments(
    n_instruments: int = 60, n_defective_target: int = 23, seed: int = 20260606
) -> list[DraftInstrument]:
    """Generate `n_instruments` synthetic draft instruments, exactly
    `n_defective_target` of which have one or more seeded defects (drawn
    with replacement from DEFECT_NAMES, up to 2 per defective instrument,
    so some instruments exercise more than one check at once), the rest
    clean by construction.
    """
    rng = random.Random(seed)
    n_clean = n_instruments - n_defective_target
    plan = ["clean"] * n_clean + ["defective"] * n_defective_target
    rng.shuffle(plan)

    instruments = []
    for i, kind in enumerate(plan):
        instrument_id = f"draft_{i:02d}"
        if kind == "clean":
            instruments.append(_build_instrument(instrument_id, [], rng))
        else:
            n_defects = 1 if rng.random() < 0.7 else 2
            defects = rng.sample(DEFECT_NAMES, k=n_defects)
            instruments.append(_build_instrument(instrument_id, defects, rng))
    return instruments
