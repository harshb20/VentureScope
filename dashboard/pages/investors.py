"""Page 6 — Investor Network.

Three charts:
  1. Investor x Sector heatmap  (investor_sector_matrix.csv + investor_summary.csv)
  2. Top investors ranking       (investor_summary.csv)
  3. Portfolio distribution      (investor_sector_matrix.csv)
"""
import numpy as np
import plotly.graph_objects as go
from dash import dcc, html

from config import COLORS, CHART_COLORS, BLUE_SCALE, apply_template

_SECTOR_COLS_DROP = {"investor_object_id", "Unknown"}


def _sector_columns(df_matrix):
    return [c for c in df_matrix.columns if c not in _SECTOR_COLS_DROP]


# --------------------------------------------------------------------------- #
# 1. Investor x Sector heatmap
# --------------------------------------------------------------------------- #
def create_investor_sector_heatmap(df_matrix, df_summary, top_n=20):
    top = df_summary.sort_values("total_investments", ascending=False).head(top_n)
    merged = top.merge(df_matrix, on="investor_object_id", how="left").fillna(0)

    sectors = _sector_columns(df_matrix)
    z = merged[sectors].to_numpy(dtype=float)
    # A few mega-investors would wash out the scale -> log1p compresses the range.
    z_log = np.log1p(z)

    fig = go.Figure(
        go.Heatmap(
            z=z_log,
            x=sectors,
            y=merged["investor_name"],
            customdata=z,
            colorscale=BLUE_SCALE,
            hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.0f} investments<extra></extra>",
            colorbar=dict(title="Deals (log)"),
        )
    )
    fig.update_layout(**apply_template(
        height=560,
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=160, r=20, t=20, b=120),
    ))
    return fig


# --------------------------------------------------------------------------- #
# 2. Top investors ranking
# --------------------------------------------------------------------------- #
def create_top_investors_chart(df_summary, top_n=15):
    df = df_summary.sort_values("total_investments", ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            y=df["investor_name"], x=df["total_investments"], orientation="h",
            marker=dict(color=COLORS["accent_purple"], opacity=0.9),
            hovertemplate="<b>%{y}</b><br>Investments: %{x:,}<extra></extra>",
        )
    )
    fig.update_layout(**apply_template(
        height=460, showlegend=False,
        xaxis=dict(title="Number of investments"),
        yaxis=dict(title=""),
        margin=dict(l=200, r=20, t=20, b=40),
    ))
    return fig


# --------------------------------------------------------------------------- #
# 3. Portfolio distribution
# --------------------------------------------------------------------------- #
def create_portfolio_distribution(df_matrix, top_n=10):
    sectors = _sector_columns(df_matrix)
    totals = df_matrix[sectors].sum().sort_values(ascending=False)
    top = totals.head(top_n)
    other = totals.iloc[top_n:].sum()

    labels = list(top.index) + (["Other"] if other > 0 else [])
    values = list(top.values) + ([other] if other > 0 else [])

    fig = go.Figure(
        go.Pie(
            labels=labels, values=values, hole=0.5,
            marker=dict(colors=(CHART_COLORS * 3)[:len(labels)]),
            textinfo="label+percent", textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:,} investments"
                          " (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(**apply_template(height=460, showlegend=False))
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
    mtx, summ = dl.investor_sector_matrix(), dl.investor_summary()
    return html.Div(
        style={"padding": "24px", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H2("Investor Network", style={"color": "#1a1d23", "marginBottom": "2px"}),
            html.P("Who backs what — the most active investors and where their capital goes.",
                   style={"color": "#5a6170", "marginTop": 0}),
            _card("Top Investors by Sector", "inv-heatmap",
                  create_investor_sector_heatmap(mtx, summ)),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px"},
                children=[
                    _card("Most Active Investors", "inv-ranking",
                          create_top_investors_chart(summ)),
                    _card("Investment Mix Across Sectors", "inv-portfolio",
                          create_portfolio_distribution(mtx)),
                ],
            ),
        ],
    )


layout = get_layout


if __name__ == "__main__":
    import dash
    app = dash.Dash(__name__)
    app.layout = get_layout()
    app.run(debug=True, host="127.0.0.1", port=8056)
