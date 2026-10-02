"""Response propensity model: of the users invited (sampling.py), who
actually completes the survey.

Propensity varies by age_band and gender (deliberately different margins
from the ones sampling.py biases on, platform and tenure, so that
nonresponse bias in this repository comes from two independent
mechanisms layered on top of each other, which is realistic: invitation
coverage and completion willingness are usually driven by different
things). This is the second, and larger, source of the gap raking.py is
built to close.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

AGE_RESPONSE_MULT = {"18-24": 0.35, "25-34": 0.70, "35-44": 1.25, "45+": 2.00}
GENDER_RESPONSE_MULT = {"woman": 1.20, "man": 0.70, "nonbinary_or_other": 0.90}
BASE_RESPONSE_PROB = 0.185


@dataclass(frozen=True)
class ResponseConfig:
    base_response_prob: float = BASE_RESPONSE_PROB
    seed: int = 20260603


def simulate_responses(
    invited_panel: pd.DataFrame, config: ResponseConfig = ResponseConfig()
) -> pd.DataFrame:
    """Among invited==True rows, simulate a binary `responded` outcome."""
    rng = np.random.default_rng(config.seed)
    age_mult = invited_panel["age_band"].map(AGE_RESPONSE_MULT).to_numpy()
    gender_mult = invited_panel["gender"].map(GENDER_RESPONSE_MULT).to_numpy()
    prob = np.clip(config.base_response_prob * age_mult * gender_mult, 0.0, 1.0)

    out = invited_panel.copy()
    out["response_prob"] = prob
    responded = np.zeros(len(out), dtype=bool)
    invited_mask = out["invited"].to_numpy()
    draws = rng.random(len(out))
    responded[invited_mask] = draws[invited_mask] < prob[invited_mask]
    out["responded"] = responded
    return out


def response_rate(responses: pd.DataFrame) -> float:
    invited = responses[responses["invited"]]
    if len(invited) == 0:
        return 0.0
    return float(invited["responded"].mean())
