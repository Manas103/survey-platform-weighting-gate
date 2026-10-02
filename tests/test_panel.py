import numpy as np

from sp.panel import FRAME_MARGINS, MARGIN_VARS, PanelConfig, simulate_panel


def test_panel_size():
    panel = simulate_panel(PanelConfig(n_users=10_000))
    assert len(panel) == 10_000


def test_panel_margins_match_frame_within_sampling_noise():
    panel = simulate_panel(PanelConfig(n_users=200_000))
    for var in MARGIN_VARS:
        observed = panel[var].value_counts(normalize=True)
        for cat, target_share in FRAME_MARGINS[var].items():
            assert abs(observed.get(cat, 0.0) - target_share) < 0.01


def test_device_tier_correlates_with_platform():
    panel = simulate_panel(PanelConfig(n_users=200_000))
    ios_high_end = (
        (panel["platform"] == "ios") & (panel["device_tier"] == "high_end")
    ).sum() / (panel["platform"] == "ios").sum()
    android_high_end = (
        (panel["platform"] == "android") & (panel["device_tier"] == "high_end")
    ).sum() / (panel["platform"] == "android").sum()
    assert ios_high_end > android_high_end
