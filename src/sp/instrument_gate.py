"""Pre-launch instrument gate: a rule-based linter that refuses a draft
survey instrument naming the exact check it failed.

Six checks, each looking for a distinct, well-known survey design defect.
This is deliberately a set of syntactic/structural heuristics, not a
semantic model of question quality: it catches the mechanical defects a
methodologist's checklist catches on a first pass, and it is validated
against a ground-truth defect label attached to each synthetic draft
instrument (see tests/test_instrument_gate.py and
scripts/run_instrument_gate.py), not against real human-authored drafts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

LEADING_PHRASES = [
    "don't you agree",
    "wouldn't you say",
    "obviously",
    "clearly,",
    "everyone knows",
]
ABSOLUTE_WORDS = ["always", "never", "completely", "totally", "every single"]
DOUBLE_BARREL_JOINERS = [" and "]
# " or " is deliberately excluded: it also appears in legitimate bipolar
# scale phrasing ("How easy or difficult...", "satisfied or dissatisfied"),
# and a first version of this check flagged those as double-barreled (see
# README, "What broke"). Classic double-barreled items join two distinct
# TOPICS, which in practice is almost always signaled by "and", not "or".


@dataclass
class SurveyItem:
    item_id: str
    text: str
    item_type: str  # "likert_agree_disagree", "likert_balanced", "grid"
    has_dont_know_option: bool = True
    scale_direction: str = "positive_first"  # or "negative_first"
    grid_row_count: int = 0


@dataclass
class DraftInstrument:
    instrument_id: str
    items: list[SurveyItem] = field(default_factory=list)
    seeded_defects: list[str] = field(default_factory=list)  # ground truth, for evaluation only


@dataclass
class GateResult:
    instrument_id: str
    passed: bool
    failed_checks: list[str]


def _check_double_barreled(item: SurveyItem) -> bool:
    text = item.text.lower()
    return any(joiner in text for joiner in DOUBLE_BARREL_JOINERS)


def _check_leading_wording(item: SurveyItem) -> bool:
    text = item.text.lower()
    return any(phrase in text for phrase in LEADING_PHRASES)


def _check_absolute_wording(item: SurveyItem) -> bool:
    text = item.text.lower()
    return any(f" {word} " in f" {text} " for word in ABSOLUTE_WORDS)


def _check_missing_dont_know(item: SurveyItem) -> bool:
    return item.item_type in ("likert_agree_disagree", "likert_balanced") and not item.has_dont_know_option


def _check_overlong_grid(item: SurveyItem, max_rows: int = 15) -> bool:
    return item.item_type == "grid" and item.grid_row_count > max_rows


CHECKS = {
    "double_barreled": _check_double_barreled,
    "leading_wording": _check_leading_wording,
    "absolute_wording": _check_absolute_wording,
    "missing_dont_know_option": _check_missing_dont_know,
    "overlong_grid": _check_overlong_grid,
}


def _check_inconsistent_scale_direction(instrument: DraftInstrument) -> bool:
    directions = {
        item.scale_direction
        for item in instrument.items
        if item.item_type in ("likert_agree_disagree", "likert_balanced")
    }
    return len(directions) > 1


def evaluate_instrument(instrument: DraftInstrument) -> GateResult:
    """Run every check against every item, plus the instrument-level
    scale-direction consistency check. Refuse (passed=False) if any check
    fails anywhere in the instrument, naming every failed check.
    """
    failed: set[str] = set()
    for item in instrument.items:
        for name, check_fn in CHECKS.items():
            if check_fn(item):
                failed.add(name)
    if _check_inconsistent_scale_direction(instrument):
        failed.add("inconsistent_scale_direction")

    return GateResult(
        instrument_id=instrument.instrument_id,
        passed=len(failed) == 0,
        failed_checks=sorted(failed),
    )
