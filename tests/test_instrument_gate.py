from sp.instrument_fixtures import generate_draft_instruments
from sp.instrument_gate import DraftInstrument, SurveyItem, evaluate_instrument


def test_clean_instrument_passes():
    instrument = DraftInstrument(
        instrument_id="clean_1",
        items=[
            SurveyItem(
                item_id="q1",
                text="How satisfied are you with your home feed?",
                item_type="likert_balanced",
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert result.passed
    assert result.failed_checks == []


def test_double_barreled_is_caught_and_named():
    instrument = DraftInstrument(
        instrument_id="d1",
        items=[
            SurveyItem(
                item_id="q1",
                text="How satisfied are you with the app's speed and its design?",
                item_type="likert_balanced",
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert not result.passed
    assert "double_barreled" in result.failed_checks


def test_legitimate_bipolar_or_phrasing_does_not_trip_double_barreled():
    """Regression test for the false positive found and fixed during
    development: "or" also appears in legitimate bipolar scale wording
    ("easy or difficult"), which must not be flagged (see README,
    "What broke").
    """
    instrument = DraftInstrument(
        instrument_id="bipolar",
        items=[
            SurveyItem(
                item_id="q1",
                text="How easy or difficult is it to find boards related to your interests?",
                item_type="likert_balanced",
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert result.passed


def test_leading_wording_is_caught():
    instrument = DraftInstrument(
        instrument_id="d2",
        items=[
            SurveyItem(
                item_id="q1",
                text="Don't you agree that the new feed design is an improvement?",
                item_type="likert_agree_disagree",
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert "leading_wording" in result.failed_checks


def test_missing_dont_know_is_caught():
    instrument = DraftInstrument(
        instrument_id="d3",
        items=[
            SurveyItem(
                item_id="q1",
                text="How useful is the new collaborative board feature?",
                item_type="likert_balanced",
                has_dont_know_option=False,
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert "missing_dont_know_option" in result.failed_checks


def test_overlong_grid_is_caught():
    instrument = DraftInstrument(
        instrument_id="d4",
        items=[
            SurveyItem(
                item_id="q1", text="Rate each feature.", item_type="grid", grid_row_count=20
            )
        ],
    )
    result = evaluate_instrument(instrument)
    assert "overlong_grid" in result.failed_checks


def test_inconsistent_scale_direction_is_caught_at_instrument_level():
    instrument = DraftInstrument(
        instrument_id="d5",
        items=[
            SurveyItem(
                item_id="q1",
                text="How positive is your experience?",
                item_type="likert_balanced",
                scale_direction="positive_first",
            ),
            SurveyItem(
                item_id="q2",
                text="How negative is your experience?",
                item_type="likert_balanced",
                scale_direction="negative_first",
            ),
        ],
    )
    result = evaluate_instrument(instrument)
    assert "inconsistent_scale_direction" in result.failed_checks


def test_fixture_generator_produces_reproducible_ground_truth_counts():
    instruments = generate_draft_instruments(n_instruments=60, n_defective_target=23)
    n_defective = sum(1 for inst in instruments if inst.seeded_defects)
    assert n_defective == 23
    assert len(instruments) == 60


def test_gate_recovers_exact_ground_truth_on_synthetic_fixtures():
    """The reference oracle for this entire module: a synthetic fixture
    generator that knows which instruments it seeded a defect into. This
    checks the gate's refusal decision against that ground truth for every
    one of the 60 fixtures, not just a handful of hand-picked examples.
    """
    instruments = generate_draft_instruments(n_instruments=60, n_defective_target=23)
    tp = fp = tn = fn = 0
    for instrument in instruments:
        result = evaluate_instrument(instrument)
        truly_defective = len(instrument.seeded_defects) > 0
        refused = not result.passed
        if truly_defective and refused:
            tp += 1
        elif truly_defective and not refused:
            fn += 1
        elif not truly_defective and refused:
            fp += 1
        else:
            tn += 1
    assert tp == 23
    assert fp == 0
    assert fn == 0
    assert tn == 37
