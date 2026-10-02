from sp.panel import PanelConfig, simulate_panel
from sp.sampling import SamplingConfig, simulate_invitations


def test_exposure_cap_bounds_invite_count():
    panel = simulate_panel(PanelConfig(n_users=20_000))
    inv = simulate_invitations(panel, SamplingConfig(exposure_cap=1))
    assert inv["invite_count"].max() <= 1


def test_invited_implies_at_least_one_trigger():
    panel = simulate_panel(PanelConfig(n_users=20_000))
    inv = simulate_invitations(panel, SamplingConfig())
    assert (inv.loc[inv["invited"], "n_triggers"] > 0).all()
    assert (inv.loc[~inv["invited"], "n_triggers"] == 0).all()


def test_more_sessions_means_higher_invite_rate():
    """iOS users session more than web users (PLATFORM_SESSION_MULT), so
    they must be invited more often, holding the per-session trigger
    probability fixed. This is the mechanism, not an incidental
    correlation, so it must hold robustly across a large panel.
    """
    panel = simulate_panel(PanelConfig(n_users=100_000))
    inv = simulate_invitations(panel, SamplingConfig())
    ios_rate = inv.loc[inv["platform"] == "ios", "invited"].mean()
    web_rate = inv.loc[inv["platform"] == "web", "invited"].mean()
    assert ios_rate > web_rate
