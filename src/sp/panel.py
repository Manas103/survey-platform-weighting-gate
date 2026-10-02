"""Simulated 200,000-user panel with known frame totals.

Six categorical variables are simulated per user. Five of them (age_band,
gender, region, platform, tenure_band) play the role of "known frame
margins", the variables a real survey team would have population totals
for from a platform-wide user table, independent of who answers any
survey. The sixth, device_tier, is deliberately held out of the raking
targets below (see raking.py and the README) so that the weighting
section has a variable to measure generalization on, not only the ones
raking is directly told to match.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

AGE_BANDS = ["18-24", "25-34", "35-44", "45+"]
GENDERS = ["woman", "man", "nonbinary_or_other"]
REGIONS = ["northeast", "south", "midwest", "west"]
PLATFORMS = ["ios", "android", "web"]
TENURE_BANDS = ["lt_6mo", "6mo_2yr", "2yr_5yr", "gt_5yr"]
DEVICE_TIERS = ["low_end", "mid_tier", "high_end"]

MARGIN_VARS = ["age_band", "gender", "region", "platform", "tenure_band"]
HOLDOUT_VAR = "device_tier"

# Frame (population) marginal probabilities. These are the "known frame
# totals" raking is calibrated against; they are a design choice standing
# in for a real platform-wide user table, not derived from anything else
# in this repository.
FRAME_MARGINS = {
    "age_band": dict(zip(AGE_BANDS, [0.22, 0.33, 0.24, 0.21])),
    "gender": dict(zip(GENDERS, [0.54, 0.43, 0.03])),
    "region": dict(zip(REGIONS, [0.18, 0.37, 0.21, 0.24])),
    "platform": dict(zip(PLATFORMS, [0.58, 0.33, 0.09])),
    "tenure_band": dict(zip(TENURE_BANDS, [0.19, 0.28, 0.27, 0.26])),
}

# device_tier's population distribution, and how strongly it correlates
# with platform (high_end skews iOS, low_end skews Android), used only to
# generate the panel and to evaluate raking's generalization, never fed to
# the raking algorithm itself.
DEVICE_TIER_BASE = dict(zip(DEVICE_TIERS, [0.30, 0.45, 0.25]))
DEVICE_TIER_PLATFORM_SHIFT = {
    "ios": {"low_end": -0.12, "mid_tier": -0.05, "high_end": 0.17},
    "android": {"low_end": 0.10, "mid_tier": 0.02, "high_end": -0.12},
    "web": {"low_end": 0.0, "mid_tier": 0.0, "high_end": 0.0},
}


def _sample_categorical(probs: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    cats = list(probs.keys())
    p = np.array(list(probs.values()))
    p = p / p.sum()
    return rng.choice(cats, size=n, p=p)


@dataclass(frozen=True)
class PanelConfig:
    n_users: int = 200_000
    seed: int = 20260601


def simulate_panel(config: PanelConfig = PanelConfig()) -> pd.DataFrame:
    """Build the simulated panel with all six categorical variables drawn
    to match FRAME_MARGINS (the five raking targets) exactly in expectation,
    plus a platform-correlated device_tier held out of raking.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_users
    data = {var: _sample_categorical(probs, n, rng) for var, probs in FRAME_MARGINS.items()}
    df = pd.DataFrame(data)

    device_tier = np.empty(n, dtype=object)
    for platform in PLATFORMS:
        mask = df["platform"].to_numpy() == platform
        n_p = int(mask.sum())
        shift = DEVICE_TIER_PLATFORM_SHIFT[platform]
        probs = {k: max(DEVICE_TIER_BASE[k] + shift[k], 0.01) for k in DEVICE_TIERS}
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        device_tier[mask] = _sample_categorical(probs, n_p, rng)
    df["device_tier"] = device_tier
    df["user_id"] = np.arange(n)
    return df


def frame_margin_table(var: str) -> pd.Series:
    return pd.Series(FRAME_MARGINS[var])
