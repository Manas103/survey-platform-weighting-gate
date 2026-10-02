"""Stratified intercept sampling with exposure caps.

A real in-product intercept survey does not sample from the panel
directly; it waits for a session, decides whether this particular session
is an eligible "intercept moment" (a probability applied per session,
not per user), and then enforces an exposure cap so a single heavy user
cannot be invited over and over inside the measurement window. Users with
more sessions get more independent chances to be the one session that
triggers an invite, which is exactly the mechanism that makes platform
and tenure correlate with invitation rate even though every session is
treated identically: it is a feature of real intercept sampling, not a
bug, and it is the first source of nonresponse bias measured in this
repository (see response.py for the second).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Mean session count over the measurement window, by platform and tenure
# band. Mobile and longer-tenured users session more often; this is a
# design choice representing realistic engagement skew, not derived from
# any other number in this repository.
PLATFORM_SESSION_MULT = {"ios": 1.35, "android": 1.0, "web": 0.30}
TENURE_SESSION_MULT = {
    "lt_6mo": 0.40,
    "6mo_2yr": 0.80,
    "2yr_5yr": 1.20,
    "gt_5yr": 1.65,
}
BASE_SESSIONS_PER_WINDOW = 6.0


@dataclass(frozen=True)
class SamplingConfig:
    trigger_prob_per_session: float = 0.05
    exposure_cap: int = 1
    seed: int = 20260602


def simulate_sessions(panel: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Poisson session counts over the measurement window, one per user."""
    platform_mult = panel["platform"].map(PLATFORM_SESSION_MULT).to_numpy()
    tenure_mult = panel["tenure_band"].map(TENURE_SESSION_MULT).to_numpy()
    lam = BASE_SESSIONS_PER_WINDOW * platform_mult * tenure_mult
    return rng.poisson(lam)


def simulate_invitations(
    panel: pd.DataFrame, config: SamplingConfig = SamplingConfig()
) -> pd.DataFrame:
    """Decide, per user, whether an intercept invitation fires.

    Each of a user's `n_sessions` sessions independently triggers at
    `trigger_prob_per_session`; a user is invited if at least one session
    triggers, but `exposure_cap` limits the invite COUNT recorded per user
    (it does not change whether they are invited at all, since every
    session still independently rolls the dice: the cap governs how many
    separate intercept instances a single user is allowed to be shown
    inside the window, which matters for exposure-cap compliance logging
    even though the simulation only needs invited-or-not downstream).
    """
    rng = np.random.default_rng(config.seed)
    n_sessions = simulate_sessions(panel, rng)
    # number of sessions that independently trigger, per user
    n_triggers = rng.binomial(n_sessions, config.trigger_prob_per_session)
    invited = n_triggers > 0
    invite_count = np.minimum(n_triggers, config.exposure_cap)

    out = panel.copy()
    out["n_sessions"] = n_sessions
    out["n_triggers"] = n_triggers
    out["invited"] = invited
    out["invite_count"] = invite_count
    return out
