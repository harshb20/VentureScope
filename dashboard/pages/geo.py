"""Page 5 — Geographic Analytics.

Three charts:
  1. Country choropleth   (country_summary.csv)
  2. City bubbles         (offices.csv)
  3. Regional comparison  (country_summary.csv, grouped to continents)

Chart builders are pure: DataFrame in, plotly Figure out.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html

from config import COLORS, BLUE_SCALE, apply_template
from dashboard.utils.chart_helpers import format_currency, format_number

# Minimal ISO-3 -> continent map (covers the funded countries; rest -> "Other").
_CONTINENT = {
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "GBR": "Europe", "DEU": "Europe", "FRA": "Europe", "ESP": "Europe", "ITA": "Europe",
    "NLD": "Europe", "SWE": "Europe", "CHE": "Europe", "IRL": "Europe", "RUS": "Europe",
    "FIN": "Europe", "DNK": "Europe", "NOR": "Europe", "BEL": "Europe", "AUT": "Europe",
    "POL": "Europe", "PRT": "Europe",
    "CHN": "Asia", "IND": "Asia", "JPN": "Asia", "ISR": "Asia", "SGP": "Asia",
    "KOR": "Asia", "HKG": "Asia", "IDN": "Asia", "TUR": "Asia", "ARE": "Asia",
    "TWN": "Asia", "MYS": "Asia", "PHL": "Asia", "THA": "Asia", "VNM": "Asia",
    "AUS": "Oceania", "NZL": "Oceania",
    "BRA": "South America", "ARG": "South America", "CHL": "South America",
    "COL": "South America",
    "ZAF": "Africa", "NGA": "Africa", "EGY": "Africa", "KEN": "Africa",
}


def _clean_countries(df):
    """Drop the aggregate 'Unknown' row that would otherwise skew the maps."""
    return df[df["country"].str.upper() != "UNKNOWN"].copy()


# --------------------------------------------------------------------------- #
# 1. Country choropleth
# --------------------------------------------------------------------------- #
def create_country_choropleth(df_country):
    df = _clean_countries(df_country)
    df = df[df["total_funding"] > 0].copy()
    # USA ($310B) dwarfs everyone -> colour on log scale, show real value in hover.
    df["log_funding"] = np.log10(df["total_funding"])

    fig = go.Figure(
        go.Choropleth(
            locations=df["country"],
            locationmode="ISO-3",
            z=df["log_funding"],
            colorscale=BLUE_SCALE,
            customdata=np.stack([df["total_funding"], df["companies"]], axis=-1),
            hovertemplate=(
                "<b>%{location}</b><br>"
                "Total funding: %{customdata[0]:$,.0f}<br>"
                "Companies: %{customdata[1]:,}<extra></extra>"
            ),
            colorbar=dict(
                title="Funding",
                tickvals=[6, 7, 8, 9, 10, 11],
                ticktext=["$1M", "$10M", "$100M", "$1B", "$10B", "$100B"],
            ),
        )
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="#e2e5ea",
        showland=True, landcolor="#f8f9fb",
        showocean=True, oceancolor="#f1f3f5",
        projection_type="natural earth",
    )
    fig.update_layout(**apply_template(height=420, margin=dict(l=0, r=0, t=0, b=0)))
    return fig


# --------------------------------------------------------------------------- #
# 2. City bubbles
# --------------------------------------------------------------------------- #
def create_city_bubbles(df_offices, top_n=300):
    df = df_offices.dropna(subset=["latitude", "longitude"]).copy()
    # ~80k rows use (0, 0) as a missing-coordinate placeholder ("null island");
    # keeping them drags every city's mean toward the Gulf of Guinea.
    df = df[(df["latitude"] != 0) | (df["longitude"] != 0)]
    df = df[df["city"].fillna("Unknown").str.lower() != "unknown"]
    g = (
        df.groupby(["city", "country_code"], as_index=False)
        .agg(companies=("object_id", "count"),
             latitude=("latitude", "mean"),
             longitude=("longitude", "mean"))
    )
    g = g.sort_values("companies", ascending=False).head(top_n)

    fig = px.scatter_geo(
        g, lat="latitude", lon="longitude",
        size="companies", size_max=26,
        color="companies", color_continuous_scale=BLUE_SCALE,
        hover_name="city",
        custom_data=["country_code", "companies"],
    )
    fig.update_traces(
        marker=dict(opacity=0.6, line=dict(width=0.5, color="#ffffff")),
        hovertemplate="<b>%{hovertext}</b>, %{customdata[0]}<br>"
                      "Companies: %{customdata[1]:,}<extra></extra>",
    )
    fig.update_geos(
        showcoastlines=True, coastlinecolor="#e2e5ea",
        showland=True, landcolor="#f8f9fb",
        showocean=True, oceancolor="#f1f3f5",
        projection_type="natural earth",
    )
    fig.update_layout(**apply_template(height=420, margin=dict(l=0, r=0, t=0, b=0)))
    return fig


# --------------------------------------------------------------------------- #
# 3. Regional comparison
# --------------------------------------------------------------------------- #
def create_regional_comparison(df_country):
    df = _clean_countries(df_country)
    df["region"] = df["country"].map(_CONTINENT).fillna("Other")
    g = (df.groupby("region", as_index=False)
           .agg(total_funding=("total_funding", "sum"),
                companies=("companies", "sum"))
           .sort_values("total_funding"))

    fig = go.Figure(
        go.Bar(
            y=g["region"], x=g["total_funding"], orientation="h",
            marker=dict(color=COLORS["accent_blue"], opacity=0.9),
            customdata=g["companies"],
            hovertemplate="<b>%{y}</b><br>Funding: %{x:$,.0f}<br>"
                          "Companies: %{customdata:,}<extra></extra>",
        )
    )
    fig.update_layout(**apply_template(
        height=360, showlegend=False,
        xaxis=dict(title="Total funding (USD)", tickprefix="$"),
        yaxis=dict(title=""),
    ))
    return fig


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
# Serve Plotly's world geometry from bundled assets so the maps render without
# reaching cdn.plot.ly (keeps the app working offline / behind firewalls).
_MAP_CONFIG = {"displayModeBar": False, "topojsonURL": "/assets/topojson/"}


def _card(title, graph_id, fig, config=None):
    return html.Div(
        className="vs-card",
        style={"background": "#fff", "border": "1px solid #e2e5ea",
               "borderRadius": "12px", "padding": "16px", "marginBottom": "16px"},
        children=[
            html.H3(title, style={"fontSize": "15px", "margin": "0 0 8px",
                                  "color": "#1a1d23"}),
            dcc.Graph(id=graph_id, figure=fig,
                      config=config or {"displayModeBar": False}),
        ],
    )


def get_layout():
    from dashboard.utils import data_loader as dl
    cs, off = dl.country_summary(), dl.offices()
    return html.Div(
        style={"padding": "24px", "maxWidth": "1200px", "margin": "0 auto"},
        children=[
            html.H2("Geographic Analytics", style={"color": "#1a1d23", "marginBottom": "2px"}),
            html.P("Where startup capital concentrates across the globe (1995–2013).",
                   style={"color": "#5a6170", "marginTop": 0}),
            _card("Funding by Country", "geo-choropleth",
                  create_country_choropleth(cs), config=_MAP_CONFIG),
            _card("Startup Hotspots by City", "geo-city-bubbles",
                  create_city_bubbles(off), config=_MAP_CONFIG),
            _card("Funding by Region", "geo-regional", create_regional_comparison(cs)),
        ],
    )


layout = get_layout


if __name__ == "__main__":
    import dash
    app = dash.Dash(__name__)
    app.layout = get_layout()
    app.run(debug=True, host="127.0.0.1", port=8055)
