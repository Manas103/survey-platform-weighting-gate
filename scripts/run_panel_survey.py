"""Build the 200,000-user panel, run stratified intercept sampling with
exposure caps, simulate responses, and measure the response rate.

Usage:
    python scripts/run_panel_survey.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sp.panel import PanelConfig, simulate_panel
from sp.response import ResponseConfig, response_rate, simulate_responses
from sp.sampling import SamplingConfig, simulate_invitations


def main() -> None:
    panel = simulate_panel(PanelConfig())
    print(f"Panel size: {len(panel)}")

    inv = simulate_invitations(panel, SamplingConfig())
    n_invited = int(inv["invited"].sum())
    print(f"Invited (stratified intercept sampling, exposure_cap=1): {n_invited} ({n_invited / len(panel):.4%} of panel)")

    resp = simulate_responses(inv, ResponseConfig())
    n_responded = int(resp["responded"].sum())
    rate = response_rate(resp)
    print(f"Responded: {n_responded}")
    print(f"RESPONSE RATE: {rate:.4%}")

    print("\nSUMMARY:")
    print(
        json.dumps(
            {
                "panel_size": len(panel),
                "n_invited": n_invited,
                "n_responded": n_responded,
                "claim_response_rate_target": 0.184,
                "measured_response_rate": rate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
