"""Run the split-ballot acquiescence experiment and report the gap
between form A (agree/disagree) and form B (balanced wording).

Usage:
    python scripts/run_split_ballot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sp.split_ballot import SplitBallotConfig, acquiescence_gap, simulate_split_ballot


def main() -> None:
    config = SplitBallotConfig()
    responses = simulate_split_ballot(config)
    result = acquiescence_gap(responses)

    print(f"n responses: {config.n_responses} (form A: {result['n_form_a']}, form B: {result['n_form_b']})")
    print(f"Form A (agree/disagree) % favorable: {result['pct_favorable_form_a']:.2f}%")
    print(f"Form B (balanced) % favorable: {result['pct_favorable_form_b']:.2f}%")
    print(f"ACQUIESCENCE GAP: {result['gap_points']:.2f} points")
    print(f"z={result['z']:.3f}, p={result['p_value']:.2e}")
    print(f"95% CI on the gap: [{result['ci95_points'][0]:.2f}, {result['ci95_points'][1]:.2f}] points")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "claim_gap_target_points": 9.3,
                "measured_gap_points": result["gap_points"],
                "p_value": result["p_value"],
                "ci95_points": result["ci95_points"],
                "ci_contains_target": result["ci95_points"][0] <= 9.3 <= result["ci95_points"][1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
