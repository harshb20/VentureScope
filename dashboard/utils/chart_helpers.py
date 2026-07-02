"""Reusable chart styling and formatting helpers."""
from config import apply_template


def style_chart(fig, title="", height=400, **overrides):
    """Apply the consistent project template to any figure."""
    fig.update_layout(**apply_template(title=title, height=height, **overrides))
    return fig


def format_currency(value):
    """Format a raw USD number as $X.XT / $X.XB / $X.XM / $XK."""
    value = float(value or 0)
    if value >= 1e12:
        return f"${value / 1e12:.1f}T"
    if value >= 1e9:
        return f"${value / 1e9:.1f}B"
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}K"
    return f"${value:,.0f}"


def format_number(value):
    """Format a large count as X.XM / X.XK / plain."""
    value = float(value or 0)
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return f"{value:,.0f}"
