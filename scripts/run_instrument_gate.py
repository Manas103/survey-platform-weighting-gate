"""Generate 60 synthetic draft instruments (23 with a seeded defect) and
run the pre-launch instrument gate against them, measuring refusal count
and precision/recall against the ground-truth seeded defects.

Usage:
    python scripts/run_instrument_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sp.instrument_fixtures import generate_draft_instruments
from sp.instrument_gate import evaluate_instrument


def main() -> None:
    instruments = generate_draft_instruments(n_instruments=60, n_defective_target=23)

    tp = fp = tn = fn = 0
    refused_lines = []
    for instrument in instruments:
        result = evaluate_instrument(instrument)
        truly_defective = len(instrument.seeded_defects) > 0
        refused = not result.passed
        if refused:
            refused_lines.append(
                f"  REFUSED {instrument.instrument_id}: failed={result.failed_checks} seeded={instrument.seeded_defects}"
            )
        if truly_defective and refused:
            tp += 1
        elif truly_defective and not refused:
            fn += 1
        elif not truly_defective and refused:
            fp += 1
        else:
            tn += 1

    print(f"{len(instruments)} draft instruments evaluated.")
    print("\n".join(refused_lines))

    n_refused = tp + fp
    print(f"\nREFUSED: {n_refused} of {len(instruments)}")
    print(f"tp={tp} fp={fp} tn={tn} fn={fn}")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"precision={precision:.3f} recall={recall:.3f}")

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "claim_refused_target": "23 of 60",
                "measured_refused": n_refused,
                "measured_total": len(instruments),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
