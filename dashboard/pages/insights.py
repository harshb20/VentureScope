"""Page 7 — Story Mode / Insights.

Three charts:
  1. Success rate donut   (companies.csv status)
  2. Pareto analysis      (sector_summary.csv)
  3. Trend sparklines     (funding_timeline.csv)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from config import COLORS, CHART_COLORS, apply_template
from dashboard.utils.chart_helpers import format_currency, format_number

# status buckets
_EXITED = {"acquired", "ipo"}
_CLOSED = {"closed"}


def _rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# --------------------------------------------------------------------------- #
# 1. Success rate donut
# --------------------------------------------------------------------------- #
def create_success_rate(df_companies):
    s = df_companies["status"].fillna("unknown").str.lower()
    exited = int(s.isin(_EXITED).sum())
    closed = int(s.isin(_CLOSED).sum())
    active = int(len(s) - exited - closed)

    # Headline success rate among companies with a decided outcome.
    decided = exited + closed
    rate = (exited / decided * 100) if decided else 0

    fig = go.Figure(
        go.Pie(
            labels=["Exited (acquired / IPO)", "Active", "Closed"],
            values=[exited, active, closed],
            hole=0.62,
            marker=dict(colors=[COLORS["accent_green"], COLORS["accent_blue"],
                                COLORS["accent_red"]]),
            sort=False,
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} companies"
                          " (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**apply_template(height=420, showlegend=False))
    fig.add_annotation(
        text=f"<b>{rate:.1f}%</b><br><span style='font-size:11px'>exit rate</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=26, color=COLORS["accent_green"]),
    )
    return fig


# --------------------------------------------------------------------------- #
# 2. Pareto analysis
# --------------------------------------------------------------------------- #
def create_pareto(df_sector, top_n=15):
    df = df_sector[df_sector["sector"].str.lower() != "unknown"].copy()
    df = df.sort_values("total_funding", ascending=False).head(top_n)
    total = df["total_funding"].sum()
    df["cum_pct"] = df["total_funding"].cumsum() / total * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=df["sector"], y=df["total_funding"],
            marker=dict(color=COLORS["accent_blue"], opacity=0.9),
            name="Funding",
            hovertemplate="<b>%{x}</b><br>Funding: %{y:$,.0f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["sector"], y=df["cum_pct"], mode="lines+markers",
            line=dict(color=COLORS["accent_amber"], width=3),
            marker=dict(size=6), name="Cumulative %",
            hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(**apply_template(
        height=440, hovermode="x unified",
        xaxis=dict(title="", tickangle=-40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=60, t=40, b=110),
    ))
    fig.update_yaxes(title_text="Total funding (USD)", tickprefix="$", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True,
                     showgrid=False)
    return fig


# --------------------------------------------------------------------------- #
# 3. Trend sparklines
# --------------------------------------------------------------------------- #
def create_trend_sparklines(df_timeline):
    df = df_timeline.sort_values("year")
    panels = [
        ("Total funding / yr", "total_funding", format_currency, COLORS["accent_blue"]),
        ("Funding rounds / yr", "funding_rounds", format_number, COLORS["accent_purple"]),
        ("Average round size", "average_round", format_currency, COLORS["accent_green"]),
    ]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.08,
    )
    for i, (title, col, fmt, color) in enumerate(panels, start=1):
        y = df[col]
        latest, prev = y.iloc[-1], y.iloc[-2]
        delta = (latest - prev) / prev * 100 if prev else 0
        fig.add_trace(
            go.Scatter(
                x=df["year"], y=y, mode="lines",
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor=_rgba(color, 0.12),
                hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=i,
        )
        fig.add_annotation(
            text=f"<b>{fmt(latest)}</b>  "
                 f"<span style='color:{COLORS['accent_green'] if delta >= 0 else COLORS['accent_red']}'>"
                 f"{'▲' if delta >= 0 else '▼'} {abs(delta):.0f}%</span>",
            xref=f"x{i if i > 1 else ''} domain", yref=f"y{i if i > 1 else ''} domain",
            x=0.02, y=1.25, showarrow=False, font=dict(size=13), align="left",
        )
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, row=1, col=i)

    fig.update_layout(**apply_template(height=240, margin=dict(l=10, r=10, t=60, b=10)))
    return fig


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
def _card(title, graph_id, fig):
    return html.Div(
        style={"background": "#fff", "border": "1px solid #e2e5ea",
               "borderRadius": "12px", "padding": "16px", "marginBottom": "16px"},
        children=[
            html.H3(title, style={"fontSize": "15px", "margin": "0 0 8px", "color": "#1a1d23"}),
            dcc.Graph(id=graph_id, figure=fig, config={"displayModeBar": False}),
        ],
    )


def get_layout():
    from dashboard.utils import data_loader as dl
    comp, sec, tl = dl.companies(), dl.sector_summary(), dl.funding_timeline()
    return html.Div(
        style={"padding": "24px", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H2("Insights", style={"color": "#1a1d23", "marginBottom": "2px"}),
            html.P("The headline patterns: exits, funding concentration, and momentum.",
                   style={"color": "#5a6170", "marginTop": 0}),
            _card("Ecosystem Momentum", "ins-sparklines", create_trend_sparklines(tl)),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    _card("Company Outcomes", "ins-success", create_success_rate(comp)),
                    _card("Funding Concentration (Pareto)", "ins-pareto",
                          create_pareto(sec)),
                ],
            ),
        ],
    )


layout = get_layout


if __name__ == "__main__":
    import dash
    app = dash.Dash(__name__)
    app.layout = get_layout()
    app.run(debug=True, host="127.0.0.1", port=8057)
