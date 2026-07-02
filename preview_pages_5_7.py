"""Standalone preview of last 3 pages's pages (5 Geo, 6 Investors, 7 Insights).

Run:  python preview_pages_5_7.py   then open http://127.0.0.1:8050
Allows to demo all three pages without the full app wired up.
"""
import dash
from dash import dcc, html, Input, Output

from dashboard.pages import geo, investors, insights

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "VentureScope — Pages 5-7 preview"

_TABS = [
    ("Geo", "/geo", geo.get_layout),
    ("Investors", "/investors", investors.get_layout),
    ("Insights", "/insights", insights.get_layout),
]

_nav = html.Div(
    style={"display": "flex", "gap": "8px", "padding": "12px 24px",
           "borderBottom": "1px solid #e2e5ea", "background": "#fff"},
    children=[dcc.Link(name, href=href,
                       style={"padding": "6px 14px", "borderRadius": "8px",
                              "textDecoration": "none", "color": "#2563eb",
                              "fontWeight": 600})
              for name, href, _ in _TABS],
)

app.layout = html.Div(
    style={"background": "#f8f9fb", "minHeight": "100vh",
           "fontFamily": "Inter, sans-serif"},
    children=[dcc.Location(id="url"), _nav, html.Div(id="page")],
)


@app.callback(Output("page", "children"), Input("url", "pathname"))
def _route(path):
    for _, href, builder in _TABS:
        if path == href:
            return builder()
    return _TABS[0][2]()  # default -> Geo


if __name__ == "__main__":
    import os
    debug = os.getenv("PREVIEW_DEBUG", "1") == "1"
    app.run(debug=debug, host="127.0.0.1", port=8050)
