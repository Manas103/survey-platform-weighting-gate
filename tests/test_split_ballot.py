import numpy as np

from sp.split_ballot import SplitBallotConfig, acquiescence_gap, simulate_split_ballot


def test_random_assignment_balances_true_favorability():
    responses = simulate_split_ballot(SplitBallotConfig(n_responses=200_000))
    a = responses[responses["form"] == "A_agree_disagree"]["true_favorable"].mean()
    b = responses[responses["form"] == "B_balanced"]["true_favorable"].mean()
    assert abs(a - b) < 0.01


def test_form_b_observed_equals_true_favorable_exactly():
    """Form B never flips: its observed rate must equal its true latent
    rate exactly, by construction, not just approximately.
    """
    responses = simulate_split_ballot(SplitBallotConfig(n_responses=5000))
    b = responses[responses["form"] == "B_balanced"]
    assert (b["observed_favorable"] == b["true_favorable"]).all()


def test_form_a_never_flips_a_favorable_response_to_unfavorable():
    responses = simulate_split_ballot(SplitBallotConfig(n_responses=5000))
    a = responses[responses["form"] == "A_agree_disagree"]
    already_favorable = a[a["true_favorable"]]
    assert (already_favorable["observed_favorable"]).all()


def test_zero_lift_gives_zero_expected_gap():
    responses = simulate_split_ballot(
        SplitBallotConfig(n_responses=200_000, acquiescence_lift=0.0)
    )
    result = acquiescence_gap(responses)
    assert abs(result["gap_points"]) < 1.0


def test_larger_lift_produces_larger_measured_gap():
    low = simulate_split_ballot(SplitBallotConfig(n_responses=200_000, acquiescence_lift=0.05, seed=1))
    high = simulate_split_ballot(SplitBallotConfig(n_responses=200_000, acquiescence_lift=0.30, seed=1))
    gap_low = acquiescence_gap(low)["gap_points"]
    gap_high = acquiescence_gap(high)["gap_points"]
    assert gap_high > gap_low


def test_gap_statistic_matches_hand_computed_z_test_oracle():
    """Independently recompute the pooled-variance two-proportion z-test by
    hand from the raw counts, diffed against acquiescence_gap's result.
    """
    responses = simulate_split_ballot(SplitBallotConfig(n_responses=4000))
    result = acquiescence_gap(responses)

    a = responses[responses["form"] == "A_agree_disagree"]["observed_favorable"]
    b = responses[responses["form"] == "B_balanced"]["observed_favorable"]
    n_a, n_b = len(a), len(b)
    x_a, x_b = a.sum(), b.sum()
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z_oracle = (x_a / n_a - x_b / n_b) / se

    assert abs(result["z"] - z_oracle) < 1e-9
