import numpy as np
import pandas as pd
import pytest

from sp.raking import (
    RakingConfig,
    average_gap_points,
    margin_gap_points,
    rake_weights,
    trim_weights,
    weighted_margin,
)


def test_untrimmed_raking_matches_target_margins_to_convergence_tolerance():
    """The defining guarantee of iterative proportional fitting: with no
    trimming, the weighted margins must match the targets to within the
    convergence tolerance, for a skewed respondent pool where every joint
    cell of the two raking variables is populated. (A degenerate case with
    an EMPTY joint cell, e.g. v1 and v2 perfectly block-sorted so no
    respondent is both "b" and "x", converges far more slowly, by design:
    see the "slow convergence on an empty joint cell" note in README,
    "What broke".)
    """
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "v1": rng.choice(["a", "b"], size=1000, p=[0.9, 0.1]),
            "v2": rng.choice(["x", "y"], size=1000, p=[0.2, 0.8]),
        }
    )
    frame_margins = {"v1": {"a": 0.5, "b": 0.5}, "v2": {"x": 0.5, "y": 0.5}}
    weights = rake_weights(df, ["v1", "v2"], frame_margins, RakingConfig(tol=1e-10))
    gap = average_gap_points(df, ["v1", "v2"], frame_margins, weights)
    assert gap < 0.01


def test_raking_converges_slowly_on_a_degenerate_empty_joint_cell():
    """Disclosed, real limitation: when two raking variables are perfectly
    confounded (every "b" is also "y", no respondent is both "b" and "x"),
    IPF still converges, but far more slowly, because fixing "b"'s margin
    and "x"'s margin independently requires an extreme, slowly-approached
    weight split within the single "a"-and-"y" group that has to carry
    both corrections. 50 iterations (the default) is not enough; 5000 is.
    """
    df = pd.DataFrame({"v1": ["a"] * 90 + ["b"] * 10, "v2": ["x"] * 20 + ["y"] * 80})
    frame_margins = {"v1": {"a": 0.5, "b": 0.5}, "v2": {"x": 0.5, "y": 0.5}}
    weights_default = rake_weights(df, ["v1", "v2"], frame_margins, RakingConfig(max_iter=50, tol=1e-10))
    gap_default = average_gap_points(df, ["v1", "v2"], frame_margins, weights_default)
    weights_many = rake_weights(df, ["v1", "v2"], frame_margins, RakingConfig(max_iter=5000, tol=1e-10))
    gap_many = average_gap_points(df, ["v1", "v2"], frame_margins, weights_many)
    assert gap_default > 0.1
    assert gap_many < 0.01


def test_raking_oracle_against_hand_computed_single_variable_case():
    """A single-variable raking problem has a closed-form answer: the
    weight for every respondent in category c must scale by
    target_share[c] / observed_share[c], independent of any other
    variable. Hand-computed for a 70/30 observed split raked to a 50/50
    target.
    """
    df = pd.DataFrame({"v1": ["a"] * 70 + ["b"] * 30})
    frame_margins = {"v1": {"a": 0.5, "b": 0.5}}
    weights = rake_weights(df, ["v1"], frame_margins, RakingConfig())

    expected_factor_a = 0.5 / 0.7
    expected_factor_b = 0.5 / 0.3
    a_weights = weights[df["v1"] == "a"]
    b_weights = weights[df["v1"] == "b"]
    assert np.allclose(a_weights, expected_factor_a, atol=1e-6)
    assert np.allclose(b_weights, expected_factor_b, atol=1e-6)


def test_weights_are_never_negative():
    df = pd.DataFrame(
        {
            "v1": np.random.RandomState(0).choice(["a", "b", "c"], size=500),
            "v2": np.random.RandomState(1).choice(["x", "y"], size=500),
        }
    )
    frame_margins = {
        "v1": {"a": 0.2, "b": 0.3, "c": 0.5},
        "v2": {"x": 0.6, "y": 0.4},
    }
    weights = rake_weights(df, ["v1", "v2"], frame_margins, RakingConfig())
    assert (weights >= 0).all()
    trimmed = trim_weights(weights, RakingConfig())
    assert (trimmed >= 0).all()


def test_trimming_sum_preserving():
    df = pd.DataFrame({"v1": np.random.RandomState(2).choice(["a", "b"], size=300)})
    frame_margins = {"v1": {"a": 0.9, "b": 0.1}}
    weights = rake_weights(df, ["v1"], frame_margins, RakingConfig())
    config = RakingConfig(trim_low=0.5, trim_high=1.5)
    trimmed = trim_weights(weights, config)
    assert abs(trimmed.sum() - weights.sum()) < 1e-6


def test_trimmed_gap_never_better_than_untrimmed_on_targeted_margins():
    """Trimming can only move weights away from the exact IPF solution, so
    the trimmed gap on the margins raking directly targets can never be
    SMALLER than the untrimmed (converged) gap, which is already near
    zero. Checked as a fuzz invariant across several trim bounds.
    """
    rng = np.random.RandomState(3)
    df = pd.DataFrame(
        {
            "v1": rng.choice(["a", "b", "c"], size=2000, p=[0.2, 0.3, 0.5]),
            "v2": rng.choice(["x", "y"], size=2000, p=[0.7, 0.3]),
        }
    )
    frame_margins = {
        "v1": {"a": 0.33, "b": 0.33, "c": 0.34},
        "v2": {"x": 0.5, "y": 0.5},
    }
    weights = rake_weights(df, ["v1", "v2"], frame_margins, RakingConfig())
    untrimmed_gap = average_gap_points(df, ["v1", "v2"], frame_margins, weights)
    for lo, hi in [(0.8, 1.2), (0.6, 1.8), (0.4, 3.0)]:
        trimmed = trim_weights(weights, RakingConfig(trim_low=lo, trim_high=hi))
        trimmed_gap = average_gap_points(df, ["v1", "v2"], frame_margins, trimmed)
        assert trimmed_gap >= untrimmed_gap - 1e-9


def test_tighter_trim_bounds_never_reduce_gap():
    rng = np.random.RandomState(4)
    df = pd.DataFrame({"v1": rng.choice(["a", "b"], size=2000, p=[0.1, 0.9])})
    frame_margins = {"v1": {"a": 0.5, "b": 0.5}}
    weights = rake_weights(df, ["v1"], frame_margins, RakingConfig())
    loose = trim_weights(weights, RakingConfig(trim_low=0.3, trim_high=4.0))
    tight = trim_weights(weights, RakingConfig(trim_low=0.8, trim_high=1.25))
    loose_gap = average_gap_points(df, ["v1"], frame_margins, loose)
    tight_gap = average_gap_points(df, ["v1"], frame_margins, tight)
    assert tight_gap >= loose_gap - 1e-9
