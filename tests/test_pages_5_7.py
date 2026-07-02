"""Smoke tests for pages 5-7: every chart builder must return a real figure
with at least one trace, using the actual processed CSVs."""
import plotly.graph_objects as go
import pytest

from dashboard.utils import data_loader as dl
from dashboard.pages import geo, investors, insights


def _assert_fig(fig):
    assert isinstance(fig, go.Figure), "builder did not return a plotly Figure"
    assert len(fig.data) >= 1, "figure has no traces"


# ---- Page 5: Geo ---------------------------------------------------------- #
def test_country_choropleth():
    _assert_fig(geo.create_country_choropleth(dl.country_summary()))


def test_city_bubbles():
    fig = geo.create_city_bubbles(dl.offices())
    _assert_fig(fig)
    # capped to the top cities, not all 112k offices
    assert len(fig.data[0].lat) <= 300


def test_regional_comparison():
    _assert_fig(geo.create_regional_comparison(dl.country_summary()))


# ---- Page 6: Investors ---------------------------------------------------- #
def test_investor_sector_heatmap():
    fig = investors.create_investor_sector_heatmap(
        dl.investor_sector_matrix(), dl.investor_summary())
    _assert_fig(fig)
    assert len(fig.data[0].y) == 20  # top-20 investors


def test_top_investors():
    _assert_fig(investors.create_top_investors_chart(dl.investor_summary()))


def test_portfolio_distribution():
    _assert_fig(investors.create_portfolio_distribution(dl.investor_sector_matrix()))


# ---- Page 7: Insights ----------------------------------------------------- #
def test_success_rate():
    _assert_fig(insights.create_success_rate(dl.companies()))


def test_pareto():
    fig = insights.create_pareto(dl.sector_summary())
    assert len(fig.data) == 2  # bar + cumulative line


def test_trend_sparklines():
    fig = insights.create_trend_sparklines(dl.funding_timeline())
    assert len(fig.data) == 3  # three sparkline panels


# ---- Page layouts build without error ------------------------------------ #
@pytest.mark.parametrize("mod", [geo, investors, insights])
def test_layout_builds(mod):
    assert mod.get_layout() is not None
