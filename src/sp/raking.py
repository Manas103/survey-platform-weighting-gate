"""Raking (iterative proportional fitting) of survey respondent weights to
known frame margins, with weight trimming.

Raking alone, run to convergence with no trimming, drives the weighted
respondent margins to match the frame margins almost exactly by
construction: that is what the algorithm is defined to do, and it is not
an interesting number to report (see README). Real survey teams trim
raking weights to a bounded range before using them, because an
untrimmed raking weight can blow up for a respondent in a thin,
badly-underrepresented cell, inflating the variance of every weighted
estimate; trimming reintroduces a small, genuine, honestly-measurable gap
against the frame totals, which is the number this repository reports.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sp.panel import FRAME_MARGINS, MARGIN_VARS


@dataclass(frozen=True)
class RakingConfig:
    max_iter: int = 50
    tol: float = 1e-8
    trim_low: float = 0.37
    trim_high: float = 3.2
    seed: int = 20260604


def rake_weights(
    respondents: pd.DataFrame,
    margin_vars: list[str] = MARGIN_VARS,
    frame_margins: dict = FRAME_MARGINS,
    config: RakingConfig = RakingConfig(),
) -> pd.Series:
    """Iterative proportional fitting: alternately rescale weights within
    each margin variable's categories so the weighted share matches the
    frame share for that variable, cycling across all `margin_vars` until
    every margin is within `tol` or `max_iter` cycles are used up.
    """
    n = len(respondents)
    weights = np.ones(n)

    for _ in range(config.max_iter):
        max_delta = 0.0
        for var in margin_vars:
            cats = respondents[var].to_numpy()
            target = frame_margins[var]
            # Snapshot the total ONCE per variable: every category of a
            # variable partitions the full respondent pool, so dividing by
            # a total that is still being updated mid-pass (recomputed
            # after each category) drifts the factors for every category
            # processed after the first and breaks convergence. This was a
            # real bug caught by the hand-computed single-variable oracle
            # test (see README, "What broke").
            variable_total = weights.sum()
            for cat, target_share in target.items():
                mask = cats == cat
                current_total = weights[mask].sum()
                if current_total <= 0:
                    continue
                current_share = current_total / variable_total
                factor = target_share / current_share if current_share > 0 else 1.0
                weights[mask] *= factor
                max_delta = max(max_delta, abs(factor - 1.0))
        if max_delta < config.tol:
            break

    return pd.Series(weights, index=respondents.index, name="raking_weight")


def trim_weights(weights: pd.Series, config: RakingConfig = RakingConfig()) -> pd.Series:
    """Cap raking weights to [trim_low, trim_high] times the mean weight,
    then renormalize so the trimmed weights still sum to the same total
    (preserving the implied respondent-equivalent sample size).
    """
    mean_w = weights.mean()
    lo, hi = config.trim_low * mean_w, config.trim_high * mean_w
    trimmed = weights.clip(lower=lo, upper=hi)
    trimmed = trimmed * (weights.sum() / trimmed.sum())
    return trimmed


def weighted_margin(
    respondents: pd.DataFrame, var: str, weights: pd.Series | None, categories: list[str]
) -> pd.Series:
    if weights is None:
        weights = pd.Series(1.0, index=respondents.index)
    w = weights.reindex(respondents.index)
    total = w.sum()
    shares = {}
    for cat in categories:
        mask = respondents[var].to_numpy() == cat
        shares[cat] = float(w[mask].sum() / total) if total > 0 else 0.0
    return pd.Series(shares)


def margin_gap_points(
    respondents: pd.DataFrame,
    var: str,
    frame_margin: dict,
    weights: pd.Series | None,
) -> float:
    """Mean absolute percentage-point gap between a weighted (or
    unweighted, if weights is None) respondent margin and the frame margin,
    across that variable's categories.
    """
    categories = list(frame_margin.keys())
    observed = weighted_margin(respondents, var, weights, categories)
    frame = pd.Series(frame_margin)
    gaps = (observed - frame).abs() * 100.0
    return float(gaps.mean())


def average_gap_points(
    respondents: pd.DataFrame,
    margin_vars: list[str],
    frame_margins: dict,
    weights: pd.Series | None,
) -> float:
    gaps = [
        margin_gap_points(respondents, var, frame_margins[var], weights)
        for var in margin_vars
    ]
    return float(np.mean(gaps))
