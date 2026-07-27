"""
CalRetail — Plotly theming.

`figure()` returns a go.Figure already wearing the app's chrome, so pages never
hand-roll layout dicts. Charts are transparent-backed: the card behind them
supplies the surface, which is what keeps light/dark switching free.
"""
from __future__ import annotations

import plotly.graph_objects as go

from frontend_dash.theme import colors

FONT = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif"


def base_layout(dark: bool = False, height: int = 240, showlegend: bool = False) -> dict:
    c = colors.chrome(dark)
    return dict(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=c["paper"],
        plot_bgcolor=c["plot"],
        font=dict(family=FONT, size=11, color=c["ink"]),
        showlegend=showlegend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=10.5, color=c["ink_muted"]),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor=c["tooltip_bg"],
            bordercolor=c["tooltip_line"],
            font=dict(family=FONT, size=11.5, color=c["tooltip_ink"]),
            align="left",
        ),
        hovermode="x unified",
        xaxis=_axis(c),
        yaxis=_axis(c, grid=True),
        dragmode=False,
        transition=dict(duration=280, easing="cubic-in-out"),
    )


def _axis(c: dict, grid: bool = False) -> dict:
    return dict(
        showgrid=grid,
        gridcolor=c["grid"],
        gridwidth=1,
        zeroline=False,
        linecolor=c["axis"],
        showline=not grid,
        ticks="",
        tickfont=dict(size=10.5, color=c["ink_muted"]),
        title=dict(font=dict(size=10.5, color=c["ink_muted"])),
        automargin=True,
    )


def figure(dark: bool = False, height: int = 240, showlegend: bool = False, **overrides) -> go.Figure:
    """A blank, fully themed figure. Pass Plotly layout kwargs to override."""
    fig = go.Figure()
    layout = base_layout(dark, height, showlegend)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


def apply(fig: go.Figure, dark: bool = False, height: int = 240,
          showlegend: bool = False, **overrides) -> go.Figure:
    """Themes a figure built elsewhere (e.g. by plotly.express)."""
    layout = base_layout(dark, height, showlegend)
    layout.update(overrides)
    fig.update_layout(**layout)
    return fig


# Plotly config used by every dcc.Graph — no modebar, no zoom, static-feeling.
GRAPH_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
    "doubleClick": False,
}


def bar_colors(values, dark: bool = False, invert: bool = False) -> list[str]:
    """Per-bar color ramp by magnitude — for ranked bar charts."""
    if not len(values):
        return []
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1
    ramp = colors.SEQUENTIAL_BLUE[2:] if not dark else colors.SEQUENTIAL_BLUE[::-1][:8]
    out = []
    for v in values:
        t = (v - lo) / span
        if invert:
            t = 1 - t
        out.append(ramp[min(int(t * (len(ramp) - 1)), len(ramp) - 1)])
    return out
