"""
VentureScope — Teammate Data Dashboard
Standalone visualization of the teammate's processed dataset.

Run:  python app.py   →  http://127.0.0.1:8060

Uses ONLY data from VentureScope-main/data/processed/.
Does not modify or depend on the original CS661 dashboard.
"""
import os
import json
import logging
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ── Logging Setup ──────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# ── Design tokens (dark theme) ─────────────────────────────────
COLORS = {
    "bg":           "#0f1117",
    "card":         "rgba(22, 24, 34, 0.85)",
    "paper":        "rgba(0,0,0,0)",
    "text":         "#f1f3f8",
    "text_sec":     "#a0a4b8",
    "text_muted":   "#6b7084",
    "border":       "rgba(255,255,255,0.06)",
    "grid":         "rgba(255,255,255,0.05)",
    "blue":         "#6366f1",
    "purple":       "#a855f7",
    "cyan":         "#22d3ee",
    "green":        "#34d399",
    "red":          "#f87171",
    "amber":        "#fbbf24",
    "pink":         "#f472b6",
}

CHART_COLORS = [
    "#6366f1", "#a855f7", "#22d3ee", "#34d399",
    "#fbbf24", "#f87171", "#f472b6", "#fb923c",
    "#2dd4bf", "#818cf8", "#c084fc", "#a3e635",
]

BLUE_SCALE = [[0, "#1e1b4b"], [0.3, "#3730a3"], [0.5, "#4f46e5"],
              [0.7, "#6366f1"], [1.0, "#a5b4fc"]]

FONT_FAMILY = "Inter, -apple-system, sans-serif"

# ── Constants ──────────────────────────────────────────────────
DEFAULT_HEIGHT = 400
MAX_TOP_ITEMS = 20
DEFAULT_PAGE_SIZE = 25

# ── Data Loader ────────────────────────────────────────────────
class DataLoader:
    """Lazy-load and cache data files."""
    
    _instance = None
    _data = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load(cls, filename: str, **kwargs) -> pd.DataFrame:
        """Load a CSV file with caching."""
        cache_key = filename
        if cache_key in cls._data:
            return cls._data[cache_key]
        
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = pd.read_csv(filepath, low_memory=False, **kwargs)
            cls._data[cache_key] = df
            logger.info(f"Loaded {filename}: {len(df):,} rows")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
    
    @classmethod
    def load_all(cls):
        """Load all required data files."""
        files = {
            "companies": "companies.csv",
            "funding": "funding_rounds.csv",
            "investors": "investors.csv",
            "offices": "offices.csv",
            "sector_summ": "sector_summary.csv",
            "country_summ": "country_summary.csv",
            "timeline": "funding_timeline.csv",
            "inv_summary": "investor_summary.csv",
            "inv_sector": "investor_sector_matrix.csv",
            "stage_year": "stage_year_trends.csv",
            "sector_year": "sector_year_trends.csv",
            "country_year": "country_year_trends.csv",
            "acquisitions": "acquisitions.csv",
        }
        
        data = {}
        for key, filename in files.items():
            data[key] = cls.load(filename)
        
        # Enrich funding data
        funding = data.get("funding", pd.DataFrame())
        if not funding.empty:
            funding["year"] = pd.to_numeric(funding.get("year", pd.Series(dtype=float)), errors="coerce")
            funding["raised_amount_usd"] = pd.to_numeric(
                funding["raised_amount_usd"], errors="coerce"
            )
        
        # Enrich companies data
        companies = data.get("companies", pd.DataFrame())
        if not companies.empty:
            companies["funding_total_usd"] = pd.to_numeric(
                companies["funding_total_usd"], errors="coerce"
            )
        
        # Year range from funding data
        years = funding["year"].dropna() if not funding.empty else pd.Series([1995, 2024])
        data["year_min"] = max(int(years.min()), 1995) if not years.empty else 1995
        data["year_max"] = int(years.max()) if not years.empty else 2024
        
        return data

# ── Data Cache ──────────────────────────────────────────────────
DATA = DataLoader.load_all()

# Extract dataframes
df_companies = DATA.get("companies", pd.DataFrame())
df_funding = DATA.get("funding", pd.DataFrame())
df_investors_raw = DATA.get("investors", pd.DataFrame())
df_offices = DATA.get("offices", pd.DataFrame())
df_sector_summ = DATA.get("sector_summ", pd.DataFrame())
df_country_summ = DATA.get("country_summ", pd.DataFrame())
df_timeline = DATA.get("timeline", pd.DataFrame())
df_inv_summary = DATA.get("inv_summary", pd.DataFrame())
df_inv_sector = DATA.get("inv_sector", pd.DataFrame())
df_stage_year = DATA.get("stage_year", pd.DataFrame())
df_sector_year = DATA.get("sector_year", pd.DataFrame())
df_country_year = DATA.get("country_year", pd.DataFrame())
df_acquisitions = DATA.get("acquisitions", pd.DataFrame())

YEAR_MIN = DATA.get("year_min", 1995)
YEAR_MAX = DATA.get("year_max", 2024)

logger.info(f"Loaded: {len(df_companies):,} companies | {len(df_funding):,} rounds | {len(df_offices):,} offices")

# ── Continent Mapping ──────────────────────────────────────────
_CONTINENT = {
    "USA": "North America", "CAN": "North America", "MEX": "North America",
    "GBR": "Europe", "DEU": "Europe", "FRA": "Europe", "ESP": "Europe",
    "ITA": "Europe", "NLD": "Europe", "SWE": "Europe", "CHE": "Europe",
    "IRL": "Europe", "RUS": "Europe", "FIN": "Europe", "DNK": "Europe",
    "NOR": "Europe", "BEL": "Europe", "AUT": "Europe", "POL": "Europe",
    "PRT": "Europe", "EST": "Europe", "ROU": "Europe", "HUN": "Europe",
    "CZE": "Europe",
    "CHN": "Asia", "IND": "Asia", "JPN": "Asia", "ISR": "Asia",
    "SGP": "Asia", "KOR": "Asia", "HKG": "Asia", "IDN": "Asia",
    "TUR": "Asia", "ARE": "Asia", "TWN": "Asia", "MYS": "Asia",
    "PHL": "Asia", "THA": "Asia", "VNM": "Asia", "PAK": "Asia",
    "BGD": "Asia",
    "AUS": "Oceania", "NZL": "Oceania",
    "BRA": "South America", "ARG": "South America", "CHL": "South America",
    "COL": "South America",
    "ZAF": "Africa", "NGA": "Africa", "EGY": "Africa", "KEN": "Africa",
}

# ── Utility Functions ──────────────────────────────────────────
def _fmt_currency(val: float) -> str:
    """Format currency values with appropriate suffix."""
    if pd.isna(val) or val == 0:
        return "$0"
    val = float(val)
    if val >= 1e12: return f"${val/1e12:.1f}T"
    if val >= 1e9:  return f"${val/1e9:.1f}B"
    if val >= 1e6:  return f"${val/1e6:.1f}M"
    if val >= 1e3:  return f"${val/1e3:.0f}K"
    return f"${val:,.0f}"

def _fmt_number(val: float) -> str:
    """Format large numbers with appropriate suffix."""
    if pd.isna(val) or val == 0:
        return "0"
    val = float(val)
    if val >= 1e6: return f"{val/1e6:.1f}M"
    if val >= 1e3: return f"{val/1e3:.1f}K"
    return f"{val:,.0f}"

def _rgba(hex_c: str, a: float) -> str:
    """Convert hex color to rgba."""
    h = hex_c.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"

def _empty_fig(msg: str, height: int = 300) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[{
            "text": msg,
            "x": 0.5,
            "y": 0.5,
            "xref": "paper",
            "yref": "paper",
            "showarrow": False,
            "font": dict(size=14, color=COLORS["text_muted"])
        }]
    )
    return fig

def apply_dark_template(height: int = DEFAULT_HEIGHT, **overrides) -> Dict:
    """Plotly layout for the dark dashboard theme."""
    layout = {
        "height": height,
        "paper_bgcolor": COLORS["paper"],
        "plot_bgcolor": COLORS["paper"],
        "font": {"family": FONT_FAMILY, "color": COLORS["text_sec"], "size": 12},
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
        "colorway": CHART_COLORS,
        "hoverlabel": {
            "bgcolor": "#1e1f2e",
            "bordercolor": COLORS["border"],
            "font": {"family": FONT_FAMILY, "color": COLORS["text"], "size": 13},
        },
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": COLORS["text_sec"]}},
        "xaxis": {
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "linecolor": COLORS["grid"],
            "title": {"font": {"color": COLORS["text_sec"]}},
        },
        "yaxis": {
            "gridcolor": COLORS["grid"],
            "zerolinecolor": COLORS["grid"],
            "linecolor": COLORS["grid"],
            "title": {"font": {"color": COLORS["text_sec"]}},
        },
    }
    # Merge overrides
    for k, v in overrides.items():
        if isinstance(v, dict) and k in layout and isinstance(layout[k], dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout

def filter_by_year(df: pd.DataFrame, year_col: str, yr_min: int, yr_max: int) -> pd.DataFrame:
    """Filter DataFrame by year range."""
    if year_col not in df.columns:
        return df
    return df[(df[year_col] >= yr_min) & (df[year_col] <= yr_max)]

# ── Dash App ──────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title="VentureScope — Teammate Dashboard",
    update_title="Loading…",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
        {"name": "description", "content": "Standalone visualization of the teammate's VentureScope dataset."},
    ],
    external_stylesheets=[dbc.themes.DARKLY],
)
server = app.server

# ── Navigation items ──────────────────────────────────────────
NAV_ITEMS = [
    {"id": "overview",  "label": "Overview",    "icon": "📊", "desc": "Command Center"},
    {"id": "geo",       "label": "Geo Analytics","icon": "🌍", "desc": "Global distribution"},
    {"id": "sectors",   "label": "Sectors",      "icon": "🏭", "desc": "Funding distribution"},
    {"id": "investors", "label": "Investors",    "icon": "💰", "desc": "Investor behavior"},
    {"id": "insights",  "label": "Insights",     "icon": "💡", "desc": "Data-driven narratives"},
    {"id": "explorer",  "label": "Explorer",     "icon": "🔍", "desc": "Browse full dataset"},
]


def create_sidebar():
    """Create the sidebar navigation."""
    nav_links = []
    for item in NAV_ITEMS:
        nav_links.append(
            html.Button(
                [
                    html.Span(item["icon"], className="nav-icon"),
                    html.Span(item["label"], className="nav-label"),
                ],
                id=f"nav-{item['id']}",
                className="nav-item",
                n_clicks=0,
                title=item["desc"],
            )
        )
    return html.Div([
        html.Div([
            html.Div("VS", className="sidebar-brand-icon"),
            html.Div([
                html.Div("VentureScope", className="sidebar-brand-text"),
                html.Div("Teammate Dashboard", className="sidebar-brand-tagline"),
            ], className="sidebar-brand-text-wrapper"),
        ], className="sidebar-brand"),
        html.Div([
            html.Div("NAVIGATION", className="sidebar-section-label"),
            *nav_links,
        ], className="sidebar-nav"),
        html.Div([
            html.Span("Source: Crunchbase via Kaggle"),
            html.Span(f"{len(df_companies):,} companies • {len(df_funding):,} rounds"),
        ], className="sidebar-footer"),
    ], id="sidebar")


def create_topbar():
    """Create the top navigation bar."""
    return html.Div([
        html.Div([
            dcc.Input(
                id="global-search",
                type="text",
                placeholder="🔍 Search companies, sectors…",
                debounce=True,
                className="topbar-search-input",
            ),
        ], className="topbar-search"),
        html.Div([
            html.Span("Year Range", className="topbar-year-label"),
            dcc.RangeSlider(
                id="global-year-range",
                min=YEAR_MIN,
                max=YEAR_MAX,
                step=1,
                value=[YEAR_MIN, YEAR_MAX],
                marks={y: str(y) for y in range(YEAR_MIN, YEAR_MAX + 1, 5)},
                tooltip={"placement": "bottom", "always_visible": False},
                className="topbar-slider",
            ),
        ], className="topbar-controls"),
        html.Div([
            html.Button(
                "⟳ Refresh",
                id="refresh-button",
                className="topbar-refresh-btn",
                n_clicks=0,
            ),
        ], className="topbar-actions"),
    ], id="topbar")


# ══════════════════════════════════════════════════════════════
#  PAGE LAYOUTS — Factory Pattern for better reusability
# ══════════════════════════════════════════════════════════════

def page_container(children, title=None, subtitle=None):
    """Wraps page content with consistent styling."""
    header = []
    if title:
        header.append(html.Div([
            html.H1(title, className="page-title"),
            html.P(subtitle or "", className="page-subtitle"),
        ], className="page-header"))
    return html.Div(header + list(children) if isinstance(children, (list, tuple)) else [children],
                    className="page-content")


def create_kpi_grid(kpis):
    """Create a grid of KPI cards."""
    return html.Div(kpis, className="kpi-grid")


def create_chart_container(title, chart, subtitle=None, controls=None, full_width=False):
    """Create a chart container with consistent styling."""
    header = html.Div([
        html.Div([
            html.Span(title, className="chart-title"),
            html.Span(subtitle or "", style={"fontSize": "11px", "color": COLORS["text_muted"]}),
        ], className="chart-header-left"),
        controls or html.Div(),
    ], className="chart-header")
    
    container_class = "chart-container chart-full" if full_width else "chart-container"
    return html.Div([header, chart], className=container_class)


# ── Overview ─────────────────────────────────────────────────
def overview_layout():
    return page_container(
        [
            html.Div(id="ov-kpis", className="kpi-grid"),
            html.Div([
                create_chart_container(
                    "Funding Velocity",
                    dcc.Graph(id="ov-velocity", config={"displayModeBar": False, "responsive": True}),
                    "Total funding per year (USD)",
                ),
                create_chart_container(
                    "Sector Allocation",
                    dcc.Graph(id="ov-sectors", config={"displayModeBar": False, "responsive": True}),
                    "Share of total funding",
                ),
            ], className="chart-row"),
            html.Div([
                create_chart_container(
                    "Global Funding Map",
                    dcc.Graph(id="ov-geo", config={"displayModeBar": False, "responsive": True}),
                    "Log₁₀ color scale",
                ),
                create_chart_container(
                    "Stage Breakdown",
                    dcc.Graph(id="ov-stages", config={"displayModeBar": False, "responsive": True}),
                    "Funding by round type",
                ),
            ], className="chart-row"),
        ],
        title="Command Center",
        subtitle="Real-time overview of the teammate's startup dataset",
    )


# ── Geo ──────────────────────────────────────────────────────
def geo_layout():
    return page_container(
        [
            html.Div([
                create_chart_container(
                    "Startup Funding by Country",
                    dcc.Graph(id="geo-choropleth", config={"displayModeBar": False, "responsive": True}),
                    "",
                    controls=html.Div([
                        dcc.Dropdown(
                            id="geo-metric",
                            options=[
                                {"label": "💰 Total Funding", "value": "total_funding"},
                                {"label": "🏢 Company Count", "value": "companies"},
                                {"label": "📊 Avg Funding", "value": "average_funding"},
                            ],
                            value="total_funding",
                            clearable=False,
                            className="geo-dropdown",
                        ),
                    ]),
                    full_width=True,
                ),
            ], className="chart-full"),
            html.Div([
                create_chart_container(
                    "Top Countries",
                    dcc.Graph(id="geo-ranking", config={"displayModeBar": False, "responsive": True}),
                ),
                create_chart_container(
                    "Startup Hotspot Cities",
                    dcc.Graph(id="geo-cities", config={"displayModeBar": False, "responsive": True}),
                    "Bubble size = number of offices",
                ),
            ], className="chart-row"),
            html.Div([
                create_chart_container(
                    "Regional Ecosystem Comparison",
                    dcc.Graph(id="geo-regions", config={"displayModeBar": False, "responsive": True}),
                    "Bars = funding, Line = companies",
                    full_width=True,
                ),
            ], className="chart-full"),
        ],
        title="Geo Analytics",
        subtitle="Global distribution of startup activity",
    )


# ── Sectors ──────────────────────────────────────────────────
def sectors_layout():
    return page_container(
        [
            html.Div(id="sec-kpis", className="kpi-grid"),
            html.Div([
                create_chart_container(
                    "Top Sectors by Funding",
                    dcc.Graph(id="sec-bars", config={"displayModeBar": False, "responsive": True}),
                    full_width=True,
                ),
            ], className="chart-full"),
            html.Div([
                create_chart_container(
                    "Sector Funding Trends",
                    dcc.Graph(id="sec-trends", config={"displayModeBar": False, "responsive": True}),
                ),
                create_chart_container(
                    "Companies per Sector",
                    dcc.Graph(id="sec-companies", config={"displayModeBar": False, "responsive": True}),
                ),
            ], className="chart-row"),
        ],
        title="Sector Analysis",
        subtitle="Funding distribution and trends across sectors",
    )


# ── Investors ────────────────────────────────────────────────
def investors_layout():
    return page_container(
        [
            html.Div(id="inv-kpis", className="kpi-grid"),
            html.Div([
                create_chart_container(
                    "Investor × Sector Heatmap",
                    dcc.Graph(id="inv-heatmap", config={"displayModeBar": False, "responsive": True}),
                    "Color = deal count (log scale)",
                    full_width=True,
                ),
            ], className="chart-full"),
            html.Div([
                create_chart_container(
                    "Most Active Investors",
                    dcc.Graph(id="inv-ranking", config={"displayModeBar": False, "responsive": True}),
                ),
                create_chart_container(
                    "Investment Mix by Sector",
                    dcc.Graph(id="inv-portfolio", config={"displayModeBar": False, "responsive": True}),
                ),
            ], className="chart-row"),
        ],
        title="Investor Intelligence",
        subtitle="Investor behavior, sector preferences, and deal patterns",
    )


# ── Insights ─────────────────────────────────────────────────
def insights_layout():
    return page_container(
        [
            html.Div(id="ins-kpis", className="kpi-grid"),
            html.Div(id="ins-cards", className="insight-grid"),
            html.Div([
                create_chart_container(
                    "Company Outcomes",
                    dcc.Graph(id="ins-success", config={"displayModeBar": False, "responsive": True}),
                ),
                create_chart_container(
                    "Funding Concentration (Pareto)",
                    dcc.Graph(id="ins-pareto", config={"displayModeBar": False, "responsive": True}),
                ),
            ], className="chart-row"),
            html.Div([
                create_chart_container(
                    "Ecosystem Momentum",
                    dcc.Graph(id="ins-sparklines", config={"displayModeBar": False, "responsive": True}),
                    "Key metrics over time",
                    full_width=True,
                ),
            ], className="chart-full"),
        ],
        title="Insights & Stories",
        subtitle="Data-driven narratives from the startup ecosystem",
    )


# ── Explorer ─────────────────────────────────────────────────
def explorer_layout():
    return page_container(
        [
            html.Div([
                # Filters sidebar
                html.Div([
                    html.Div("🔍 Filters", className="explorer-filters-title"),
                    html.Div([
                        html.Div("STATUS", className="filter-label"),
                        dcc.Dropdown(
                            id="exp-status",
                            options=[
                                {"label": s.title(), "value": s}
                                for s in sorted(df_companies["status"].dropna().unique())
                                if s
                            ],
                            multi=True,
                            placeholder="All statuses",
                            className="filter-dropdown",
                        ),
                    ]),
                    html.Div([
                        html.Div("SECTOR", className="filter-label"),
                        dcc.Dropdown(
                            id="exp-sector",
                            options=[
                                {"label": s.title(), "value": s}
                                for s in sorted(df_companies["sector"].dropna().unique())
                                if s and s.lower() != "unknown"
                            ][:40],
                            multi=True,
                            placeholder="All sectors",
                            className="filter-dropdown",
                        ),
                    ]),
                    html.Div([
                        html.Div("COUNTRY", className="filter-label"),
                        dcc.Dropdown(
                            id="exp-country",
                            options=[
                                {"label": c, "value": c}
                                for c in sorted(df_companies["country"].dropna().unique())
                                if c and c != "Unknown"
                            ][:60],
                            multi=True,
                            placeholder="All countries",
                            className="filter-dropdown",
                        ),
                    ]),
                    html.Div([
                        html.Div("FUNDING RANGE", className="filter-label"),
                        dcc.RangeSlider(
                            id="exp-funding-range",
                            min=0,
                            max=int(df_companies["funding_total_usd"].max() or 1e9),
                            step=100000,
                            value=[0, int(df_companies["funding_total_usd"].max() or 1e9)],
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="filter-slider",
                        ),
                    ]),
                    html.Button(
                        "Clear All Filters",
                        id="exp-clear-filters",
                        className="filter-clear-btn",
                        n_clicks=0,
                    ),
                ], className="explorer-filters"),
                # Results
                html.Div([
                    html.Div(id="exp-count", className="explorer-count"),
                    html.Div(id="exp-table", className="explorer-table-wrapper"),
                ], className="explorer-results"),
            ], className="explorer-container"),
        ],
        title="Company Explorer",
        subtitle="Browse and search the full dataset",
    )


# ══════════════════════════════════════════════════════════════
#  APP LAYOUT
# ══════════════════════════════════════════════════════════════
app.layout = html.Div([
    dcc.Store(id="current-page", data="overview"),
    dcc.Store(id="filter-state", data={}),
    html.Div([
        create_sidebar(),
        html.Div([
            create_topbar(),
            html.Div(id="page-content"),
        ], id="main-content"),
    ], id="app-container"),
])


# ══════════════════════════════════════════════════════════════
#  CALLBACKS — Optimized with lru_cache where applicable
# ══════════════════════════════════════════════════════════════

# ── Navigation ────────────────────────────────────────────────
@app.callback(
    [Output("page-content", "children"), Output("current-page", "data")]
    + [Output(f"nav-{item['id']}", "className") for item in NAV_ITEMS],
    [Input(f"nav-{item['id']}", "n_clicks") for item in NAV_ITEMS],
    [State("current-page", "data")],
    prevent_initial_call=False,
)
def navigate(*args):
    """Handle navigation between pages."""
    clicks = args[:len(NAV_ITEMS)]
    current_page = args[-1]
    ctx = dash.callback_context
    
    if ctx.triggered and ctx.triggered[0]["prop_id"] != ".":
        page_id = ctx.triggered[0]["prop_id"].split(".")[0].replace("nav-", "")
    else:
        page_id = current_page or "overview"
    
    page_map = {
        "overview": overview_layout,
        "geo": geo_layout,
        "sectors": sectors_layout,
        "investors": investors_layout,
        "insights": insights_layout,
        "explorer": explorer_layout,
    }
    content = page_map.get(page_id, overview_layout)()
    nav_classes = ["nav-item active" if item["id"] == page_id else "nav-item" for item in NAV_ITEMS]
    return [content, page_id] + nav_classes


# ── Overview Callback ─────────────────────────────────────────
@app.callback(
    [Output("ov-kpis", "children"), Output("ov-velocity", "figure"),
     Output("ov-sectors", "figure"), Output("ov-geo", "figure"),
     Output("ov-stages", "figure")],
    [Input("current-page", "data"), Input("global-year-range", "value")],
)
def update_overview(page: str, yr: List[int]) -> Tuple[List, go.Figure, go.Figure, go.Figure, go.Figure]:
    """Update the overview page with KPI cards and charts."""
    if page != "overview":
        return [], go.Figure(), go.Figure(), go.Figure(), go.Figure()
    
    yr_min, yr_max = yr or [YEAR_MIN, YEAR_MAX]
    fr = filter_by_year(df_funding, "year", yr_min, yr_max)
    
    total_funding = fr["raised_amount_usd"].sum()
    total_deals = len(fr)
    total_companies = len(df_companies)
    disclosed = fr[fr["raised_amount_usd"] > 0]
    avg_deal = disclosed["raised_amount_usd"].mean() if len(disclosed) else 0
    
    # KPI Cards
    kpis = [
        html.Div([
            html.Div("💰 TOTAL FUNDING", className="kpi-label"),
            html.Div(_fmt_currency(total_funding), className="kpi-value"),
            html.Div(f"{total_deals:,} rounds", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("🏢 COMPANIES", className="kpi-label"),
            html.Div(_fmt_number(total_companies), className="kpi-value"),
            html.Div("in dataset", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📊 AVG ROUND SIZE", className="kpi-label"),
            html.Div(_fmt_currency(avg_deal), className="kpi-value"),
            html.Div("per round", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📈 UNIQUE SECTORS", className="kpi-label"),
            html.Div(str(fr["funding_round_type"].nunique() if "funding_round_type" in fr.columns else "—"),
                      className="kpi-value"),
            html.Div(f"{yr_min}–{yr_max}", className="kpi-change positive"),
        ], className="kpi-card"),
    ]
    
    # Velocity chart
    tl = fr.groupby("year").agg(
        total=("raised_amount_usd", "sum"),
        count=("funding_round_id", "count")
    ).reset_index().sort_values("year")
    
    vel = go.Figure(go.Scatter(
        x=tl["year"], y=tl["total"], mode="lines+markers",
        fill="tozeroy", line=dict(color=COLORS["blue"], width=2.5),
        marker=dict(size=5, color=COLORS["blue"]),
        fillcolor=_rgba(COLORS["blue"], 0.1),
        hovertemplate="<b>%{x:.0f}</b><br>$%{y:,.0f}<extra></extra>",
    ))
    vel.update_layout(**apply_dark_template(height=320, showlegend=False,
        xaxis_title="Year", yaxis_title="Total Funding (USD)", hovermode="x unified"))
    
    # Sector donut
    if "funding_round_type" in fr.columns:
        sec = fr.groupby("funding_round_type")["raised_amount_usd"].sum().reset_index()
        sec = sec.sort_values("raised_amount_usd", ascending=False).head(10)
        sec_fig = go.Figure(go.Pie(
            labels=sec["funding_round_type"], values=sec["raised_amount_usd"],
            hole=0.55, sort=True,
            marker=dict(colors=CHART_COLORS[:len(sec)], line=dict(color=COLORS["bg"], width=2)),
            textinfo="percent", textposition="inside",
            textfont=dict(size=11, color="#ffffff"),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        sec_fig.update_layout(**apply_dark_template(height=320, showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=0.65, font=dict(size=11)),
            margin=dict(l=10, r=10, t=20, b=20)))
    else:
        sec_fig = _empty_fig("No sector data", 320)
    
    # Geo choropleth
    cs = df_country_summ[df_country_summ["total_funding"] > 0].copy()
    geo_fig = go.Figure(go.Choropleth(
        locations=cs["country"], z=np.log10(cs["total_funding"].clip(lower=1)),
        colorscale=BLUE_SCALE, showscale=True, locationmode="ISO-3",
        customdata=np.stack([cs["total_funding"], cs["companies"]], axis=-1),
        hovertemplate="<b>%{location}</b><br>Funding: $%{customdata[0]:,.0f}<br>Companies: %{customdata[1]:,}<extra></extra>",
        colorbar=dict(title=dict(text="Log₁₀($)", font=dict(size=11, color=COLORS["text_sec"])),
                      tickfont=dict(color=COLORS["text_muted"], size=10), len=0.6, thickness=12),
        marker_line_color=COLORS["border"], marker_line_width=0.5,
    ))
    geo_fig.update_layout(**apply_dark_template(height=320, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#1a1c2e", showframe=False,
                 showcoastlines=True, coastlinecolor="#2a2d42", projection_type="natural earth",
                 showocean=True, oceancolor="#0d0f1a", lakecolor="#141628"))
    
    # Stage bar
    if "funding_round_type" in fr.columns:
        stg = fr.groupby("funding_round_type")["raised_amount_usd"].sum().reset_index()
        stg = stg.sort_values("raised_amount_usd", ascending=True)
        stg_fig = go.Figure(go.Bar(
            x=stg["raised_amount_usd"], y=stg["funding_round_type"], orientation="h",
            marker=dict(color=stg["raised_amount_usd"],
                        colorscale=[[0, COLORS["purple"]], [1, COLORS["blue"]]]),
            hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
        ))
        stg_fig.update_layout(**apply_dark_template(height=320, showlegend=False, xaxis_title="Total Funding (USD)"))
    else:
        stg_fig = _empty_fig("No stage data", 320)
    
    return [kpis, vel, sec_fig, geo_fig, stg_fig]


# ── Geo Callback ──────────────────────────────────────────────
@app.callback(
    [Output("geo-choropleth", "figure"), Output("geo-ranking", "figure"),
     Output("geo-cities", "figure"), Output("geo-regions", "figure")],
    [Input("current-page", "data"), Input("global-year-range", "value"),
     Input("geo-metric", "value")],
)
def update_geo(page: str, yr: List[int], metric: str) -> Tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """Update the Geo Analytics page."""
    if page != "geo":
        return [go.Figure()] * 4
    
    yr_min, yr_max = yr or [YEAR_MIN, YEAR_MAX]
    metric = metric or "total_funding"
    
    cs = df_country_summ.copy()
    cs = cs[cs["country"].str.upper() != "UNKNOWN"]
    cs = cs[cs["total_funding"] > 0]
    
    z_col = metric
    z_vals = cs[z_col].clip(lower=1)
    labels = {"total_funding": "Total Funding", "companies": "Companies", "average_funding": "Avg Funding"}
    
    # Choropleth
    choro = go.Figure(go.Choropleth(
        locations=cs["country"], z=np.log10(z_vals), locationmode="ISO-3",
        colorscale=BLUE_SCALE, showscale=True,
        customdata=np.stack([cs["total_funding"], cs["companies"], cs["average_funding"]], axis=-1),
        hovertemplate="<b>%{location}</b><br>Funding: $%{customdata[0]:,.0f}<br>Companies: %{customdata[1]:,}<br>Avg: $%{customdata[2]:,.0f}<extra></extra>",
        colorbar=dict(title=dict(text=f"Log₁₀({labels.get(z_col,'')})", font=dict(size=11, color=COLORS["text_sec"])),
                      tickfont=dict(color=COLORS["text_muted"], size=10), len=0.7, thickness=12),
        marker_line_color="#2a2d42", marker_line_width=0.5,
    ))
    choro.update_layout(**apply_dark_template(height=450, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#1a1c2e", showframe=False,
                 showcoastlines=True, coastlinecolor="#2a2d42", projection_type="natural earth",
                 showocean=True, oceancolor="#0d0f1a", lakecolor="#141628"))
    
    # Country ranking
    top = cs.nlargest(15, z_col).sort_values(z_col, ascending=True)
    rank = go.Figure(go.Bar(
        x=top[z_col], y=top["country"], orientation="h",
        marker=dict(color=top[z_col],
                    colorscale=[[0, COLORS["cyan"]], [1, COLORS["blue"]]]),
        hovertemplate="<b>%{y}</b><br>%{x:,.0f}<extra></extra>",
    ))
    rank.update_layout(**apply_dark_template(height=400, showlegend=False, xaxis_title=labels.get(z_col, "")))
    
    # City bubbles
    off = df_offices.dropna(subset=["latitude", "longitude"]).copy()
    off = off[(off["latitude"] != 0) | (off["longitude"] != 0)]
    off = off[off["city"].fillna("Unknown").str.lower() != "unknown"]
    cg = off.groupby(["city", "country_code"], as_index=False).agg(
        companies=("object_id", "count"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean")
    )
    cg = cg.nlargest(200, "companies")
    
    city_fig = go.Figure(go.Scattergeo(
        lat=cg["latitude"], lon=cg["longitude"],
        text=cg.apply(lambda r: f"{r['city']}, {r['country_code']}", axis=1),
        marker=dict(size=np.sqrt(cg["companies"]) * 2.5, color=cg["companies"],
                    colorscale=[[0, _rgba(COLORS["blue"], 0.3)], [1, _rgba(COLORS["cyan"], 0.85)]],
                    line=dict(width=0.5, color=_rgba(COLORS["blue"], 0.4)), sizemode="diameter",
                    colorbar=dict(title=dict(text="Offices", font=dict(size=11, color=COLORS["text_sec"])),
                                  tickfont=dict(color=COLORS["text_muted"], size=10), len=0.5, thickness=10)),
        customdata=cg["companies"],
        hovertemplate="<b>%{text}</b><br>Offices: %{customdata:,}<extra></extra>",
    ))
    city_fig.update_layout(**apply_dark_template(height=400, margin=dict(l=0, r=0, t=10, b=0)),
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#1a1c2e", showframe=False,
                 showcoastlines=True, coastlinecolor="#2a2d42", projection_type="natural earth",
                 showocean=True, oceancolor="#0d0f1a"))
    
    # Regional comparison
    cs2 = df_country_summ.copy()
    cs2 = cs2[cs2["country"].str.upper() != "UNKNOWN"]
    cs2["region"] = cs2["country"].map(_CONTINENT).fillna("Other")
    rg = cs2.groupby("region", as_index=False).agg(
        total_funding=("total_funding", "sum"),
        companies=("companies", "sum")
    ).sort_values("total_funding", ascending=False)
    
    reg = go.Figure()
    reg.add_trace(go.Bar(
        x=rg["region"], y=rg["total_funding"], name="Total Funding ($)",
        marker=dict(color=COLORS["blue"]),
        hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>"
    ))
    reg.add_trace(go.Scatter(
        x=rg["region"], y=rg["companies"], name="Companies",
        mode="lines+markers", line=dict(color=COLORS["amber"], width=2),
        marker=dict(size=8), yaxis="y2",
        hovertemplate="<b>%{x}</b><br>%{y:,} companies<extra></extra>"
    ))
    reg.update_layout(**apply_dark_template(height=350,
        yaxis=dict(title="Total Funding (USD)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis2=dict(title="Company Count", overlaying="y", side="right", showgrid=False,
                    titlefont=dict(color=COLORS["amber"]), tickfont=dict(color=COLORS["amber"]))
    ))
    
    return [choro, rank, city_fig, reg]


# ── Sectors Callback ──────────────────────────────────────────
@app.callback(
    [Output("sec-kpis", "children"), Output("sec-bars", "figure"),
     Output("sec-trends", "figure"), Output("sec-companies", "figure")],
    [Input("current-page", "data"), Input("global-year-range", "value")],
)
def update_sectors(page: str, yr: List[int]) -> Tuple[List, go.Figure, go.Figure, go.Figure]:
    """Update the Sectors page."""
    if page != "sectors":
        return [], go.Figure(), go.Figure(), go.Figure()
    
    yr_min, yr_max = yr or [YEAR_MIN, YEAR_MAX]
    
    ss = df_sector_summ[df_sector_summ["sector"].str.lower() != "unknown"].copy()
    ss = ss.sort_values("total_funding", ascending=False)
    top_sec = ss.iloc[0]["sector"] if len(ss) > 0 else "—"
    total_sec = len(ss)
    top_funded = ss.iloc[0]["total_funding"] if len(ss) > 0 else 0
    
    kpis = [
        html.Div([
            html.Div("🏆 TOP SECTOR", className="kpi-label"),
            html.Div(top_sec.title()[:16], className="kpi-value", style={"fontSize": "20px"}),
            html.Div(_fmt_currency(top_funded), className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📂 TOTAL SECTORS", className="kpi-label"),
            html.Div(str(total_sec), className="kpi-value"),
            html.Div("tracked", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("🏢 TOTAL COMPANIES", className="kpi-label"),
            html.Div(_fmt_number(ss["companies"].sum()), className="kpi-value"),
            html.Div("across sectors", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("💰 TOTAL FUNDING", className="kpi-label"),
            html.Div(_fmt_currency(ss["total_funding"].sum()), className="kpi-value"),
            html.Div("all sectors", className="kpi-change positive"),
        ], className="kpi-card"),
    ]
    
    # Top sectors bar
    top15 = ss.head(15).sort_values("total_funding", ascending=True)
    bars = go.Figure(go.Bar(
        x=top15["total_funding"], y=top15["sector"].str.title(), orientation="h",
        marker=dict(color=top15["total_funding"],
                    colorscale=[[0, COLORS["purple"]], [1, COLORS["blue"]]]),
        hovertemplate="<b>%{y}</b><br>$%{x:,.0f}<extra></extra>",
    ))
    bars.update_layout(**apply_dark_template(height=480, showlegend=False, xaxis_title="Total Funding (USD)"))
    
    # Sector year trends (top 6)
    sy = df_sector_year.copy()
    sy = sy[sy["sector"].str.lower() != "unknown"]
    top_sectors = sy.groupby("sector")["total_funding"].sum().nlargest(6).index.tolist()
    sy_top = sy[sy["sector"].isin(top_sectors)]
    
    trends = go.Figure()
    for i, sec_name in enumerate(top_sectors):
        d = sy_top[sy_top["sector"] == sec_name].sort_values("year")
        trends.add_trace(go.Scatter(
            x=d["year"], y=d["total_funding"], name=sec_name.title(),
            mode="lines", line=dict(width=2, color=CHART_COLORS[i % len(CHART_COLORS)]),
            hovertemplate=f"<b>{sec_name.title()}</b><br>%{{x:.0f}}: $%{{y:,.0f}}<extra></extra>",
        ))
    trends.update_layout(**apply_dark_template(height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Year", yaxis_title="Total Funding (USD)", hovermode="x unified"))
    
    # Companies per sector
    comp_sec = ss.head(15).sort_values("companies", ascending=True)
    comp_fig = go.Figure(go.Bar(
        x=comp_sec["companies"], y=comp_sec["sector"].str.title(), orientation="h",
        marker=dict(color=comp_sec["companies"],
                    colorscale=[[0, COLORS["green"]], [1, COLORS["cyan"]]]),
        hovertemplate="<b>%{y}</b><br>%{x:,} companies<extra></extra>",
    ))
    comp_fig.update_layout(**apply_dark_template(height=380, showlegend=False, xaxis_title="Company Count"))
    
    return [kpis, bars, trends, comp_fig]


# ── Investors Callback ────────────────────────────────────────
@app.callback(
    [Output("inv-kpis", "children"), Output("inv-heatmap", "figure"),
     Output("inv-ranking", "figure"), Output("inv-portfolio", "figure")],
    [Input("current-page", "data"), Input("global-year-range", "value")],
)
def update_investors(page: str, yr: List[int]) -> Tuple[List, go.Figure, go.Figure, go.Figure]:
    """Update the Investors page."""
    if page != "investors":
        return [], go.Figure(), go.Figure(), go.Figure()
    
    total_investors = len(df_inv_summary)
    top_inv = df_inv_summary.sort_values("total_investments", ascending=False).iloc[0] if len(df_inv_summary) > 0 else None
    
    kpis = [
        html.Div([
            html.Div("👥 TOTAL INVESTORS", className="kpi-label"),
            html.Div(f"{total_investors:,}", className="kpi-value"),
            html.Div("unique", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("🏆 TOP INVESTOR", className="kpi-label"),
            html.Div(str(top_inv["investor_name"])[:18] if top_inv is not None else "—",
                     className="kpi-value", style={"fontSize": "18px"}),
            html.Div(f"{int(top_inv['total_investments']):,} deals" if top_inv is not None else "—",
                     className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📊 TOTAL DEALS", className="kpi-label"),
            html.Div(_fmt_number(df_inv_summary["total_investments"].sum()), className="kpi-value"),
            html.Div("tracked", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📈 AVG DEALS/INVESTOR", className="kpi-label"),
            html.Div(f"{df_inv_summary['total_investments'].mean():.1f}", className="kpi-value"),
            html.Div("deals avg", className="kpi-change positive"),
        ], className="kpi-card"),
    ]
    
    # Heatmap
    _SKIP = {"investor_object_id", "Unknown"}
    sector_cols = [c for c in df_inv_sector.columns if c not in _SKIP]
    top20 = df_inv_summary.sort_values("total_investments", ascending=False).head(20)
    merged = top20.merge(df_inv_sector, on="investor_object_id", how="left").fillna(0)
    z = merged[sector_cols].to_numpy(dtype=float)
    z_log = np.log1p(z)
    
    hm = go.Figure(go.Heatmap(
        z=z_log, x=[s.title() for s in sector_cols], y=merged["investor_name"],
        customdata=z, colorscale=BLUE_SCALE,
        hovertemplate="<b>%{y}</b><br>%{x}: %{customdata:.0f} deals<extra></extra>",
        colorbar=dict(title="Deals (log)", titlefont=dict(color=COLORS["text_sec"], size=11),
                      tickfont=dict(color=COLORS["text_muted"])),
    ))
    hm.update_layout(**apply_dark_template(height=560,
        xaxis=dict(title="", tickangle=-45), yaxis=dict(title="", autorange="reversed"),
        margin=dict(l=180, r=20, t=20, b=120)))
    
    # Top investors ranking
    top15 = df_inv_summary.sort_values("total_investments", ascending=True).tail(15)
    rank = go.Figure(go.Bar(
        y=top15["investor_name"], x=top15["total_investments"], orientation="h",
        marker=dict(color=top15["total_investments"],
                    colorscale=[[0, COLORS["purple"]], [1, COLORS["blue"]]]),
        hovertemplate="<b>%{y}</b><br>%{x:,} deals<extra></extra>",
    ))
    rank.update_layout(**apply_dark_template(height=460, showlegend=False,
        xaxis_title="Total Investments", margin=dict(l=200, r=20, t=20, b=40)))
    
    # Portfolio distribution
    totals = df_inv_sector[sector_cols].sum().sort_values(ascending=False)
    top10 = totals.head(10)
    other = totals.iloc[10:].sum()
    labels = [s.title() for s in top10.index] + (["Other"] if other > 0 else [])
    values = list(top10.values) + ([other] if other > 0 else [])
    
    port = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=(CHART_COLORS * 3)[:len(labels)], line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+percent", textposition="outside",
        textfont=dict(color=COLORS["text_sec"], size=10),
        hovertemplate="<b>%{label}</b><br>%{value:,} deals (%{percent})<extra></extra>",
    ))
    port.update_layout(**apply_dark_template(height=460, showlegend=False))
    
    return [kpis, hm, rank, port]


# ── Insights Callback ─────────────────────────────────────────
@app.callback(
    [Output("ins-kpis", "children"), Output("ins-cards", "children"),
     Output("ins-success", "figure"), Output("ins-pareto", "figure"),
     Output("ins-sparklines", "figure")],
    [Input("current-page", "data"), Input("global-year-range", "value")],
)
def update_insights(page: str, yr: List[int]) -> Tuple[List, List, go.Figure, go.Figure, go.Figure]:
    """Update the Insights page."""
    if page != "insights":
        return [], [], go.Figure(), go.Figure(), go.Figure()
    
    s = df_companies["status"].fillna("unknown").str.lower()
    _EXITED = {"acquired", "ipo"}
    _CLOSED = {"closed"}
    exited = int(s.isin(_EXITED).sum())
    closed = int(s.isin(_CLOSED).sum())
    active = len(s) - exited - closed
    decided = exited + closed
    rate = (exited / decided * 100) if decided else 0
    
    total = len(df_companies)
    funded = df_companies[df_companies["funding_total_usd"] > 0]
    total_funding = funded["funding_total_usd"].sum()
    
    ss = df_sector_summ[df_sector_summ["sector"].str.lower() != "unknown"].sort_values("total_funding", ascending=False)
    top_sec = ss.iloc[0]["sector"].title() if len(ss) > 0 else "—"
    
    kpis = [
        html.Div([
            html.Div("🎯 EXIT RATE", className="kpi-label"),
            html.Div(f"{rate:.1f}%", className="kpi-value"),
            html.Div(f"{exited:,} exits", className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("📉 CLOSURE RATE", className="kpi-label"),
            html.Div(f"{closed/total*100:.1f}%", className="kpi-value"),
            html.Div(f"{closed:,} closed", className="kpi-change negative"),
        ], className="kpi-card"),
        html.Div([
            html.Div("🏭 DOMINANT SECTOR", className="kpi-label"),
            html.Div(top_sec[:16], className="kpi-value", style={"fontSize": "18px"}),
            html.Div(_fmt_currency(ss.iloc[0]["total_funding"]) if len(ss) > 0 else "—",
                     className="kpi-change positive"),
        ], className="kpi-card"),
        html.Div([
            html.Div("💰 TOTAL FUNDED", className="kpi-label"),
            html.Div(_fmt_currency(total_funding), className="kpi-value"),
            html.Div(f"{len(funded):,} companies", className="kpi-change positive"),
        ], className="kpi-card"),
    ]
    
    # Insight cards
    def _card(title, body, tags, color):
        border_colors = {"blue": COLORS["blue"], "purple": COLORS["purple"],
                         "green": COLORS["green"], "amber": COLORS["amber"]}
        return html.Div([
            html.Div(title, className="insight-title"),
            html.Div(body, className="insight-body"),
            html.Div([html.Span(t, className=f"insight-tag insight-tag-{color}") for t in tags],
                     style={"marginTop": "12px"}),
        ], className="insight-card", style={"borderLeftColor": border_colors.get(color, COLORS["blue"])})
    
    top3 = ss.head(3)
    top3_pct = top3["total_funding"].sum() / total_funding * 100 if total_funding > 0 else 0
    
    cards = [
        _card(
            "🌍 Ecosystem Scale",
            f"The dataset tracks {total:,} companies across {len(ss)} sectors, "
            f"with {_fmt_currency(total_funding)} in total tracked funding. "
            f"This makes it one of the most comprehensive startup datasets available.",
            ["Scale", "Coverage"], "blue"
        ),
        _card(
            "📊 Sector Concentration",
            f"The top 3 sectors ({', '.join(top3['sector'].str.title())}) "
            f"capture {top3_pct:.0f}% of all funding — a significant concentration risk.",
            ["Sectors", "Risk"], "purple"
        ),
        _card(
            "🚀 Exit Landscape",
            f"Only {rate:.1f}% of startups achieve exit (acquisition or IPO), "
            f"while {closed/total*100:.1f}% close. The remaining {active/total*100:.0f}% are still operating.",
            ["Exits", "Returns"], "green"
        ),
        _card(
            "📈 Power Law of Returns",
            f"Startup funding follows a power law: the top-funded companies "
            f"command a disproportionate share of capital. Most raise modest rounds.",
            ["Funding", "Distribution"], "amber"
        ),
    ]
    
    # Success rate donut
    succ = go.Figure(go.Pie(
        labels=["Exited (Acquired/IPO)", "Active", "Closed"],
        values=[exited, active, closed], hole=0.62, sort=False,
        marker=dict(colors=[COLORS["green"], COLORS["blue"], COLORS["red"]],
                    line=dict(color=COLORS["bg"], width=2)),
        textinfo="label+percent",
        textfont=dict(color=COLORS["text_sec"], size=10),
        hovertemplate="<b>%{label}</b><br>%{value:,} companies (%{percent})<extra></extra>",
    ))
    succ.update_layout(**apply_dark_template(height=400, showlegend=False))
    succ.add_annotation(text=f"<b>{rate:.1f}%</b><br><span style='font-size:11px'>exit rate</span>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=26, color=COLORS["green"]))
    
    # Pareto chart
    if len(funded) > 0:
        fs = funded.sort_values("funding_total_usd", ascending=False).reset_index(drop=True)
        fs["cum_pct"] = fs["funding_total_usd"].cumsum() / total_funding * 100
        fs["comp_pct"] = (fs.index + 1) / len(fs) * 100
        pareto = go.Figure()
        pareto.add_trace(go.Scatter(
            x=fs["comp_pct"], y=fs["cum_pct"], mode="lines", fill="tozeroy",
            line=dict(color=COLORS["blue"], width=2),
            fillcolor=_rgba(COLORS["blue"], 0.08),
            name="Actual", hovertemplate="Top %{x:.0f}% → %{y:.0f}% of funding<extra></extra>"
        ))
        pareto.add_trace(go.Scatter(
            x=[0, 100], y=[0, 100], mode="lines",
            line=dict(color=COLORS["text_muted"], dash="dash", width=1),
            name="Perfect Equality", hoverinfo="skip"
        ))
    else:
        pareto = _empty_fig("No funded companies", 400)
    
    pareto.update_layout(**apply_dark_template(height=400, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="% of Companies", yaxis_title="% of Total Funding"))
    
    # Sparklines
    tl = df_timeline.sort_values("year")
    panels = [
        ("💰 Total Funding / yr", "total_funding", _fmt_currency, COLORS["blue"]),
        ("📊 Rounds / yr", "funding_rounds", _fmt_number, COLORS["purple"]),
        ("📈 Avg Round Size", "average_round", _fmt_currency, COLORS["green"]),
    ]
    spark = make_subplots(rows=1, cols=3, subplot_titles=[p[0] for p in panels],
                          horizontal_spacing=0.08)
    
    for ann in spark.layout.annotations:
        ann.font = dict(color=COLORS["text_sec"], size=13, family=FONT_FAMILY)
    
    for i, (title, col, fmt, color) in enumerate(panels, 1):
        y_vals = tl[col]
        if len(y_vals) >= 2:
            latest, prev = y_vals.iloc[-1], y_vals.iloc[-2]
            delta = (latest - prev) / prev * 100 if prev else 0
        else:
            latest = y_vals.iloc[-1] if len(y_vals) > 0 else 0
            delta = 0
        spark.add_trace(go.Scatter(
            x=tl["year"], y=y_vals, mode="lines",
            line=dict(color=color, width=2.5),
            fill="tozeroy", fillcolor=_rgba(color, 0.1),
            hovertemplate="%{x:.0f}: %{y:,.0f}<extra></extra>", showlegend=False,
        ), row=1, col=i)
        spark.add_annotation(
            text=f"<b>{fmt(latest)}</b>  <span style='color:{COLORS['green'] if delta >= 0 else COLORS['red']}'>"
                 f"{'▲' if delta >= 0 else '▼'} {abs(delta):.0f}%</span>",
            xref=f"x{i if i > 1 else ''} domain", yref=f"y{i if i > 1 else ''} domain",
            x=0.02, y=1.3, showarrow=False, font=dict(size=13), align="left"
        )
        spark.update_xaxes(visible=False, row=1, col=i)
        spark.update_yaxes(visible=False, row=1, col=i)
    
    spark.update_layout(**apply_dark_template(height=240, margin=dict(l=10, r=10, t=70, b=10)))
    
    return [kpis, cards, succ, pareto, spark]


# ── Explorer Callback ─────────────────────────────────────────
@app.callback(
    [Output("exp-count", "children"), Output("exp-table", "children")],
    [Input("current-page", "data"),
     Input("exp-status", "value"),
     Input("exp-sector", "value"),
     Input("exp-country", "value"),
     Input("global-search", "value")],
)
def update_explorer(page: str, status: List[str], sector: List[str],
                    country: List[str], search: str) -> Tuple[List, dash_table.DataTable]:
    """Update the Explorer page with filtered data."""
    if page != "explorer":
        return ["", ""]
    
    df = df_companies.copy()
    
    if status:
        df = df[df["status"].isin(status)]
    if sector:
        df = df[df["sector"].isin(sector)]
    if country:
        df = df[df["country"].isin(country)]
    if search:
        mask = df["company_name"].fillna("").str.contains(search, case=False, na=False)
        df = df[mask]
    
    count_text = [
        html.Span("Showing "),
        html.Strong(f"{min(len(df), 200):,}"),
        html.Span(f" of {len(df):,} companies")
    ]
    
    display_cols = ["company_name", "sector", "country", "status", "funding_total_usd", "funding_rounds"]
    display = df[display_cols].head(200).copy()
    display.columns = ["Company", "Sector", "Country", "Status", "Total Funding ($)", "Rounds"]
    
    # Format currency in the table
    display["Total Funding ($)"] = display["Total Funding ($)"].apply(
        lambda x: _fmt_currency(x) if pd.notna(x) else "$0"
    )
    
    table = dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        page_size=DEFAULT_PAGE_SIZE,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "minHeight": "400px"},
        style_header={
            "backgroundColor": "rgba(255,255,255,0.03)",
            "color": COLORS["text_sec"],
            "fontWeight": "600",
            "fontSize": "12px",
            "textTransform": "uppercase",
            "letterSpacing": "0.6px",
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "12px 16px",
        },
        style_cell={
            "backgroundColor": "transparent",
            "color": COLORS["text"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "fontSize": "13px",
            "padding": "10px 16px",
            "fontFamily": FONT_FAMILY,
            "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "rgba(255,255,255,0.02)"},
            {"if": {"state": "active"}, "backgroundColor": COLORS["card"], "border": "none"},
        ],
        style_cell_conditional=[
            {"if": {"column_id": "Company"}, "fontWeight": "500", "color": COLORS["blue"]},
            {"if": {"column_id": "Total Funding ($)"}, "fontFamily": "var(--font-mono)"},
        ],
    )
    return [count_text, table]


# ── Explorer Clear Filters ─────────────────────────────────────
@app.callback(
    [Output("exp-status", "value"), Output("exp-sector", "value"),
     Output("exp-country", "value"), Output("global-search", "value")],
    [Input("exp-clear-filters", "n_clicks")],
    prevent_initial_call=True,
)
def clear_filters(n_clicks: int) -> Tuple[None, None, None, str]:
    """Clear all filters in the Explorer page."""
    if n_clicks > 0:
        return None, None, None, ""
    return None, None, None, ""


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n🚀 VentureScope Teammate Dashboard")
    print(f"   http://127.0.0.1:8060")
    print(f"   {len(df_companies):,} companies | {len(df_funding):,} rounds | {len(df_offices):,} offices\n")
    app.run(host="127.0.0.1", port=8060, debug=True)
