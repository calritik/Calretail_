"""
CalRetail — reusable UI primitives.

Every page composes from these so the whole console stays visually identical.
Class names map 1:1 onto assets/style.css; nothing here inlines styling that
the stylesheet already owns (that's what keeps dark mode free).
"""
from __future__ import annotations

import re

from dash import dcc, html

from frontend_dash.theme import colors
from frontend_dash.theme.chart_theme import GRAPH_CONFIG

# ── Icons ────────────────────────────────────────────────────────────────────
# dash.html carries no SVG elements, so every icon is a CSS mask over
# currentColor (see the .ic-* rules in assets/style.css). That keeps icons
# crisp, theme-aware and dependency-free.

def icon(name: str, cls: str = ""):
    """An icon span. `name` matches an .ic-<name> rule in style.css."""
    return html.Span(className=f"ic ic-{name} {cls}".strip())


NAV_ICONS = {k: icon(k) for k in
             ("home", "cx", "merch", "ops", "support", "monet", "assistant")}


# ── Cards ────────────────────────────────────────────────────────────────────

_BOLD = re.compile(r"</?b>")


def rich(text: str):
    """
    Renders a string that uses <b>…</b> for emphasis.

    Dash escapes strings, so passing markup straight through would print the
    literal tags. Splitting on the tag and rebuilding with html.B keeps the
    emphasis without ever handing raw HTML to the DOM — which also means an
    unbalanced tag degrades to plain text instead of breaking the layout.
    """
    if not text or "<b>" not in text:
        return text
    parts = _BOLD.split(text)
    # split() alternates outside/inside, because the tags alternate open/close.
    return [p if i % 2 == 0 else html.B(p) for i, p in enumerate(parts) if p]


def card(title, body, caption: str = "", info: str = "", span: int = 1,
         cls: str = "", **kw):
    """
    The standard console card.

    title   — heading text (or a component)
    body    — the card's content
    caption — muted one-liner under the body, explaining what the reader sees
    info    — text for the hover "i" tooltip (the *how it works* detail)
    span    — 2 makes the card span both grid columns
    """
    children = []
    if info:
        children.append(
            html.Div(
                html.Span(["i", html.Span(rich(info), className="info-pop")], className="info-ic"),
                className="info-corner",
            )
        )
    children.append(
        html.Div(html.Div(html.Span(title), className="card-head"), className="card-title")
    )
    children.append(html.Div(body, className="card-pad"))
    if caption:
        children.append(html.Div(caption, className="card-caption"))

    classes = "card" + (" span-2" if span == 2 else "") + (f" {cls}" if cls else "")
    return html.Article(children, className=classes, **kw)


def split(left, right, weight: str = "", ruled: bool = True):
    """
    Lays a card's body out as two columns side by side.

    A card carrying KPIs, a chart and two tables is several screens tall when
    everything is stacked, and by the time the reader reaches the bottom the
    heading that framed it is long gone. Splitting turns that tall card into a
    wide one.

    The split is driven by a container query on the card, not by the viewport
    (see .card-split in style.css), so this is safe to use anywhere: a card in a
    half-width grid column keeps both halves stacked, and the same call splits
    once the card spans the full grid. Pages don't have to reason about width.

    left / right — the two columns; either may be a list of components.
    weight       — "" for equal halves, "wide-left" or "wide-right" for ~60/40.
    ruled        — draw the hairline between the columns.
    """
    cls = "card-split"
    if weight:
        cls += f" split-{weight}"
    if ruled:
        cls += " split-ruled"
    return html.Div([html.Div(left), html.Div(right)], className=cls)


def subhead(text):
    """A small uppercase label introducing a block inside a card."""
    return html.Div(text, className="card-sub mt-14")


# ── Small primitives ─────────────────────────────────────────────────────────

def money(v, exact_below: float = 100000) -> str:
    """
    Indian-notation rupees, scaled to whichever unit reads cleanly.

    Retail figures here span five orders of magnitude — a ₹107 sock and a ₹12Cr
    CLV portfolio — so one fixed unit always mangles an end of the range
    ("₹0.29Cr" costs the reader a beat that "₹29.0L" doesn't). Below a lakh the
    exact figure is shortest, so it's used as-is.
    """
    v = v or 0
    a = abs(v)
    if a < exact_below:
        return f"₹{v:,.0f}"
    if round(a / 100000, 1) < 100:
        return f"₹{v / 100000:.1f}L"
    return f"₹{v / 10000000:.2f}Cr"


def pill(text, level: str = "neutral"):
    """Status pill. level: high | medium | low | info | neutral (or a raw
    backend label — risk_class maps it)."""
    lvl = level if level in {"high", "medium", "low", "info", "neutral"} else colors.risk_class(level)
    return html.Span([html.Span(className="pdot"), text], className=f"pill {lvl}")


def kpi(label, value, delta: str = "", trend: str = ""):
    """A single stat tile. trend: 'up' | 'down' | '' — colors the delta line."""
    kids = [html.Div(label, className="kpi-k"), html.Div(value, className="kpi-v")]
    if delta:
        kids.append(html.Div(delta, className=f"kpi-d {trend}".strip()))
    return html.Div(kids, className="kpi")


def kpi_grid(items):
    """items: list of kpi() nodes."""
    return html.Div(items, className="kpi-grid")


def bar_row(label, value_text, pct: float, tone: str = ""):
    """
    Labelled progress bar. pct is 0-100; tone: '' | ok | warn | danger.

    Passing an empty label drops the label column entirely rather than leaving
    a blank track — the aspect-analysis table puts these inside a narrow cell,
    where reserving space for a label that isn't there squeezed the bar.
    """
    pct = max(0.0, min(100.0, float(pct or 0)))
    return html.Div(
        [
            html.Span(label, className="bar-row-k"),
            html.Div(
                html.Div(className=f"bar-fill {tone}".strip(), style={"width": f"{pct:.1f}%"}),
                className="bar-track",
            ),
            html.Span(value_text, className="bar-row-v tabular"),
        ],
        className="bar-row" if label else "bar-row nolabel",
    )


def stat_row(label, value):
    return html.Div([html.Span(label, className="stat-k"), html.Span(value, className="stat-v")],
                    className="stat-row")


def stat_list(pairs):
    """
    pairs: [(label, value), ...]

    Six rows fit a card without pushing it past the fold; beyond that the
    list scrolls internally rather than stretching the card taller, since a
    card with a chart above it and ten stat rows below was routinely running
    past a full viewport.
    """
    cls = "stat-list-scroll" if len(pairs) > 6 else ""
    return html.Div([stat_row(k, v) for k, v in pairs], className=cls)


def table(headers, rows, numeric: set[int] | None = None,
          wide: set[int] | None = None, narrow: set[int] | None = None,
          full: bool = False):
    """
    Compact data table.

    headers — list of column labels
    rows    — list of row lists (cells may be strings or components)
    numeric — indexes to right-align, tabular-align and keep on one line
    wide    — indexes that carry the substance (a product name, an issue
              summary) and should be given a generous share of the width
    narrow  — indexes holding a short code or pill, shrunk to their content
    full    — the table is the card's whole point, so it gets the card's height
              instead of the 232px scroll well

    The columns size themselves (table-layout: auto), so `wide`/`narrow` are
    hints for the cases where the browser can't tell that a column of 4-letter
    status pills doesn't deserve the same width as a column of product names.
    """
    numeric = numeric or set()
    wide = wide or set()
    narrow = narrow or set()

    def _cls(i: int, extra: str = "") -> str:
        parts = [extra] if extra else []
        if i in numeric:
            parts.append("num")
        if i in wide:
            parts.append("col-wide")
        if i in narrow:
            parts.append("col-narrow")
        return " ".join(parts)

    head = html.Tr([html.Th(h, className=_cls(i)) for i, h in enumerate(headers)])
    body = [
        html.Tr([html.Td(c, className=_cls(i, "tabular" if i in numeric else ""))
                 for i, c in enumerate(r)])
        for r in rows
    ]
    return html.Div(html.Table([html.Thead(head), html.Tbody(body)], className="tbl"),
                    className="tbl-wrap tbl-full" if full else "tbl-wrap")


def graph(figure, height: int = 240, **kw):
    return dcc.Graph(figure=figure, config=GRAPH_CONFIG,
                     style={"height": f"{height}px"}, **kw)


def empty(text="No data available."):
    return html.Div(text, className="cp-empty")


def offline_banner():
    return html.Div(
        [
            html.Span("⚠"),
            html.Span("Backend not responding — start it with "),
            html.Code("myenv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000"),
        ],
        className="banner err",
    )


def caveat(text):
    """
    A card-level caveat about what the data can and cannot support.

    Not a "this is fake" marker — nothing on the console is generated. This is
    for genuine analytical limits worth stating on the card itself: a metric a
    join can't split, a proxy standing in for telemetry the dataset lacks, or a
    model whose measured lift is nil. Those belong in front of the reader, not
    buried in a tooltip.
    """
    return html.Div([html.Span("◆ "), html.Span(text)], className="banner warn",
                    style={"marginBottom": "12px", "fontSize": "11.5px", "padding": "8px 12px"})
