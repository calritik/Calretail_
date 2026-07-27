"""
CalRetail — app shell.

Layout mirrors the Industrial-AI portfolio console: content column on the left,
a sticky navigation rail on the right, and a page header carrying an eyebrow /
title / subtitle triplet that animates on every route change.
"""
from __future__ import annotations

from dash import dcc, html

from frontend_dash.components.cards import NAV_ICONS

# (path, icon key, short index label, nav label)
NAV_ITEMS = [
    ("/",                     "home",      "Overview",  "AI Portfolio"),
    ("/customer-experience",  "cx",        "Domain 01", "Customer Experience"),
    ("/merchandising",        "merch",     "Domain 02", "Merchandising"),
    ("/operations",           "ops",       "Domain 03", "Operational Efficiency"),
    ("/support",              "support",   "Domain 04", "Customer Support"),
    ("/ai-assistant",         "assistant", "Copilot",   "AI Assistant"),
]


def _sun_moon():
    """Both glyphs ship; CSS shows whichever matches the active theme."""
    return [html.Span(className="ic ic-sun"), html.Span(className="ic ic-moon")]


def sidebar(active_path: str = "/"):
    links = []
    for path, ikey, short, label in NAV_ITEMS:
        active = path == active_path
        links.append(
            dcc.Link(
                [
                    html.Span(NAV_ICONS.get(ikey), className="nav-ic"),
                    html.Span([html.Span(short, className="nav-index"),
                               html.Span(label, className="nav-label")], className="nav-txt"),
                    html.Span("›", className="nav-caret"),
                ],
                href=path,
                className="nav-btn" + (" active" if active else ""),
            )
        )

    return html.Aside(
        [
            html.Div(
                [
                    html.Div("CR", className="brand-mark"),
                    html.Div([html.Div("CalRetail", className="brand-name"),
                              html.Div("Retail AI Console", className="brand-sub")]),
                ],
                className="sidebar-brand",
            ),
            # No footer status box: backend health and the "30/30 live" count
            # already live in the "Portfolio at a glance" KPI card on the home
            # page — repeating them here just duplicated the same two facts in
            # a second place a click away.
            html.Nav([html.Div("Domains", className="nav-heading")] + links, className="nav-scroll"),
        ],
        className="sidebar",
    )


def page_header(eyebrow: str, title: str, subtitle: str):
    return html.Header(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(eyebrow, className="eyebrow"),
                        html.H1(title, className="page-title title-swap"),
                        html.P(subtitle, className="page-subtitle title-swap"),
                    ]
                ),
                html.Button(_sun_moon(), id="theme-toggle", className="theme-btn",
                            n_clicks=0, title="Toggle dark mode", **{"aria-label": "Toggle dark mode"}),
            ],
            className="page-head-row",
        ),
        className="page-head",
    )


def module_page(eyebrow: str, title: str, subtitle: str, children):
    """A full page: animated header + staggered-entrance content grid."""
    return html.Div(
        [
            page_header(eyebrow, title, subtitle),
            html.Section(children, className="module-view module-enter"),
        ]
    )


def app_shell(active_path: str, content):
    return html.Div(
        [html.Main(content, className="main-col"), sidebar(active_path)],
        className="app-shell",
    )
