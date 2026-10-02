"""Measure the average margin gap against known frame totals, unweighted,
raking-only (untrimmed), and raking-plus-trimming, on the full simulated
panel and response pipeline. Also reports the gap on device_tier, a
variable deliberately held out of the raking targets.

Usage:
    python scripts/run_raking.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sp.panel import (
    DEVICE_TIER_BASE,
    FRAME_MARGINS,
    HOLDOUT_VAR,
    MARGIN_VARS,
    PanelConfig,
    simulate_panel,
)
from sp.raking import RakingConfig, average_gap_points, margin_gap_points, rake_weights, trim_weights
from sp.response import ResponseConfig, simulate_responses
from sp.sampling import SamplingConfig, simulate_invitations


def main() -> None:
    panel = simulate_panel(PanelConfig())
    inv = simulate_invitations(panel, SamplingConfig())
    resp = simulate_responses(inv, ResponseConfig())
    respondents = resp[resp["responded"]].copy()
    print(f"n respondents: {len(respondents)}")

    unweighted_gap = average_gap_points(respondents, MARGIN_VARS, FRAME_MARGINS, None)
    print(f"\nUnweighted average gap vs frame totals (5 margins): {unweighted_gap:.3f} points")

    config = RakingConfig()
    weights = rake_weights(respondents, MARGIN_VARS, FRAME_MARGINS, config)
    untrimmed_gap = average_gap_points(respondents, MARGIN_VARS, FRAME_MARGINS, weights)
    print(f"Raked, UNTRIMMED, average gap: {untrimmed_gap:.8f} points (near-exact by IPF's own guarantee, not the reported claim)")

    trimmed_weights = trim_weights(weights, config)
    trimmed_gap = average_gap_points(respondents, MARGIN_VARS, FRAME_MARGINS, trimmed_weights)
    print(f"Raked AND TRIMMED (trim_low={config.trim_low}, trim_high={config.trim_high}), average gap: {trimmed_gap:.3f} points")

    print("\nPer-margin gaps (unweighted vs trimmed):")
    for var in MARGIN_VARS:
        g_un = margin_gap_points(respondents, var, FRAME_MARGINS[var], None)
        g_tr = margin_gap_points(respondents, var, FRAME_MARGINS[var], trimmed_weights)
        print(f"  {var}: unweighted={g_un:.3f}  trimmed={g_tr:.3f}")

    holdout_unweighted = margin_gap_points(respondents, HOLDOUT_VAR, DEVICE_TIER_BASE, None)
    holdout_trimmed = margin_gap_points(respondents, HOLDOUT_VAR, DEVICE_TIER_BASE, trimmed_weights)
    print(f"\nHoldout variable '{HOLDOUT_VAR}' (NOT a raking target) gap: unweighted={holdout_unweighted:.3f}  trimmed={holdout_trimmed:.3f}")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "claim_before_target": 7.9,
                "measured_unweighted_gap": unweighted_gap,
                "claim_after_target": 1.2,
                "measured_trimmed_gap": trimmed_gap,
                "untrimmed_raked_gap_near_zero": untrimmed_gap,
                "holdout_device_tier_gap_unweighted": holdout_unweighted,
                "holdout_device_tier_gap_trimmed": holdout_trimmed,
                "trim_low": config.trim_low,
                "trim_high": config.trim_high,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
