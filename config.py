"""
VentureScope — configuration & design tokens.

⚠️ PLACEHOLDER SHIM (owned by the Integration/UI team).
This file exists so pages 5–7 (geo/investors/insights) can render and be
previewed standalone. When the real config.py lands, delete this and keep the
same public names: COLORS, CHART_COLORS, apply_template().
"""

# Brand palette — must stay identical across all 7 pages.
COLORS = {
    "accent_blue": "#2563eb",
    "accent_purple": "#7c3aed",
    "accent_cyan": "#0891b2",
    "accent_green": "#059669",
    "accent_red": "#dc2626",
    "accent_amber": "#d97706",
}

CHART_COLORS = [
    "#2563eb",  # Blue
    "#7c3aed",  # Purple
    "#0891b2",  # Cyan
    "#059669",  # Green
    "#d97706",  # Amber
    "#dc2626",  # Red
]

# Sequential blue ramp reused by the choropleth / heatmap.
BLUE_SCALE = [[0.0, "#eef2ff"], [0.5, "#3b82f6"], [1.0, "#1e3a8a"]]

FONT_FAMILY = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"
INK = "#1a1d23"
MUTED = "#5a6170"
GRID = "#e2e5ea"
PAPER = "#ffffff"


def apply_template(title="", height=400, **overrides):
    """Return a Plotly layout kwargs dict with the project's light theme.

    Any keyword in ``overrides`` (e.g. xaxis=..., margin=..., showlegend=...)
    is merged on top, so callers can customise per chart.
    """
    layout = dict(
        title=dict(text=title, font=dict(size=16, color=INK, family=FONT_FAMILY)),
        height=height,
        paper_bgcolor=PAPER,
        plot_bgcolor=PAPER,
        font=dict(family=FONT_FAMILY, color=MUTED, size=12),
        margin=dict(l=40, r=20, t=48 if title else 20, b=40),
        colorway=CHART_COLORS,
        hoverlabel=dict(font=dict(family=FONT_FAMILY, size=12)),
    )
    # Give axes a light grid unless the caller overrides them.
    axis_style = dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                      title=dict(font=dict(color=MUTED)))
    layout["xaxis"] = {**axis_style}
    layout["yaxis"] = {**axis_style}

    for key, val in overrides.items():
        if key in ("xaxis", "yaxis") and isinstance(val, dict):
            layout[key] = {**layout[key], **val}
        else:
            layout[key] = val
    return layout
