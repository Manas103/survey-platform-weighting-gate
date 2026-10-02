"""Split-ballot experiment isolating an acquiescence gap between two
wordings of the same underlying construct.

Form A uses an agree/disagree ("Do you agree that...") frame on a single
construct, which is the classic setup for acquiescence bias: some
respondents tend to agree with whatever is asked regardless of content.
Form B asks the identical construct with a balanced, non-agreement frame
("How [positive adjective] or [negative adjective] do you find..."), which
does not offer a default "yes" to lean on. Respondents are randomly
assigned to one form; any difference in the measured percent-favorable
between forms, on the same underlying construct, is attributable to
wording, not to a real difference in attitude, because assignment is
random.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

TRUE_FAVORABLE_RATE = 0.52
ACQUIESCENCE_LIFT = 0.194
ACQUIESCENT_RESPONDENT_SHARE = 1.0  # see README: lift is applied uniformly


@dataclass(frozen=True)
class SplitBallotConfig:
    n_responses: int = 4000
    true_favorable_rate: float = TRUE_FAVORABLE_RATE
    acquiescence_lift: float = ACQUIESCENCE_LIFT
    seed: int = 20260605


def simulate_split_ballot(config: SplitBallotConfig = SplitBallotConfig()) -> pd.DataFrame:
    """Randomly assign each of `n_responses` simulated respondents to form
    A (agree/disagree) or form B (balanced), draw their TRUE latent
    favorability toward the construct once (independent of form), then
    draw the OBSERVED favorable/unfavorable response: form B's observed
    response equals the true latent favorability; form A's observed
    response additionally flips an unfavorable latent response to
    favorable at rate `acquiescence_lift`, modeling a respondent who leans
    toward "agree" regardless of their true view.
    """
    rng = np.random.default_rng(config.seed)
    n = config.n_responses
    form = rng.choice(["A_agree_disagree", "B_balanced"], size=n)
    true_favorable = rng.random(n) < config.true_favorable_rate

    observed_favorable = true_favorable.copy()
    is_form_a = form == "A_agree_disagree"
    unfavorable_on_a = is_form_a & (~true_favorable)
    flips = rng.random(n) < config.acquiescence_lift
    observed_favorable[unfavorable_on_a & flips] = True

    return pd.DataFrame(
        {
            "form": form,
            "true_favorable": true_favorable,
            "observed_favorable": observed_favorable,
        }
    )


def acquiescence_gap(responses: pd.DataFrame) -> dict:
    """Two-sample z-test for proportions: percent favorable on form A minus
    percent favorable on form B, with a 95% CI on the difference.
    """
    a = responses[responses["form"] == "A_agree_disagree"]["observed_favorable"]
    b = responses[responses["form"] == "B_balanced"]["observed_favorable"]
    n_a, n_b = len(a), len(b)
    p_a, p_b = a.mean(), b.mean()
    gap = p_a - p_b

    p_pool = (a.sum() + b.sum()) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = gap / se_pool if se_pool > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = gap - 1.96 * se_unpooled
    ci_high = gap + 1.96 * se_unpooled

    return {
        "n_form_a": int(n_a),
        "n_form_b": int(n_b),
        "pct_favorable_form_a": float(p_a * 100),
        "pct_favorable_form_b": float(p_b * 100),
        "gap_points": float(gap * 100),
        "z": float(z),
        "p_value": float(p_value),
        "ci95_points": (float(ci_low * 100), float(ci_high * 100)),
    }
