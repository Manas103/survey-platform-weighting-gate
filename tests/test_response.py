from sp.panel import PanelConfig, simulate_panel
from sp.response import ResponseConfig, response_rate, simulate_responses
from sp.sampling import SamplingConfig, simulate_invitations


def test_only_invited_can_respond():
    panel = simulate_panel(PanelConfig(n_users=20_000))
    inv = simulate_invitations(panel, SamplingConfig())
    resp = simulate_responses(inv, ResponseConfig())
    assert (resp.loc[~resp["invited"], "responded"] == False).all()  # noqa: E712


def test_older_age_band_responds_more_than_youngest():
    panel = simulate_panel(PanelConfig(n_users=150_000))
    inv = simulate_invitations(panel, SamplingConfig())
    resp = simulate_responses(inv, ResponseConfig())
    invited = resp[resp["invited"]]
    young = invited[invited["age_band"] == "18-24"]["responded"].mean()
    old = invited[invited["age_band"] == "45+"]["responded"].mean()
    assert old > young


def test_response_rate_is_between_zero_and_one():
    panel = simulate_panel(PanelConfig(n_users=20_000))
    inv = simulate_invitations(panel, SamplingConfig())
    resp = simulate_responses(inv, ResponseConfig())
    rate = response_rate(resp)
    assert 0.0 < rate < 1.0
