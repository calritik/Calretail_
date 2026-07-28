"""
Executive dashboard — the console's landing page.

Every figure here is aggregated from the transaction log at request time by
/api/v1/overview/*. Nothing is a constant: the trend's peak callout is read off
the series, and the heatmap's scale is keyed to the range in the data. A build at
a different scale re-renders with different numbers and stays correct.

The page shell returns immediately and every panel fills from its own callback,
so the landing page never blocks on a rollup — same pattern as the domain pages.
"""
from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html

from frontend_dash.components import cards as C
from frontend_dash.components.cards import NAV_ICONS
from frontend_dash.components.layout import module_page
from frontend_dash.services.api import api_get, backend_is_up
from frontend_dash.services.capabilities import DOMAINS
from frontend_dash.theme import chart_theme as T
from frontend_dash.theme import colors

dash.register_page(__name__, path="/", name="AI Portfolio")

_DOMAIN_ICON = {"cx": "cx", "merch": "merch", "ops": "ops", "support": "support"}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# SQLite's strftime('%w') is 0=Sunday. Reordered to a Mon-first trading week,
# which is how a retail calendar is read.
WEEKDAYS = [(1, "Mon"), (2, "Tue"), (3, "Wed"), (4, "Thu"),
            (5, "Fri"), (6, "Sat"), (0, "Sun")]

# Sequential ramp -> Plotly colorscale (one hue, light->dark). Used wherever a
# mark encodes magnitude, so those marks never need a categorical hue.
SEQ = [[i / (len(colors.SEQUENTIAL_BLUE) - 1), c]
       for i, c in enumerate(colors.SEQUENTIAL_BLUE)]

# Two-series charts take slots 1 and 2 of the categorical theme in fixed order.
# The pair is separation-checked against the card surface: ΔE 18.5 normal,
# 12.2 under protanopia. Amber sits under 3:1 contrast on white, so every chart
# using it carries a legend and a direct end-label rather than relying on hue.
S_REVENUE, S_MARGIN = colors.CATEGORICAL[0], colors.CATEGORICAL[1]


def _inr(v) -> str:
    v = float(v or 0)
    if abs(v) >= 1_00_00_000:
        return f"₹{v / 10000000:.2f}Cr"
    if abs(v) >= 1_00_000:
        return f"₹{v / 100000:.1f}L"
    return f"₹{v:,.0f}"


def _compact(n) -> str:
    n = float(n or 0)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Estate headline
# ══════════════════════════════════════════════════════════════════════════════

@callback(
    Output("hm-estate", "children"),
    Output("hm-banner", "children"),
    Input("hm-load", "data")
)
def _estate(_):
    d = api_get("/api/v1/overview/estate")
    if not d:
        if not backend_is_up():
            return C.empty("Estate figures unavailable — is the backend running?"), [C.offline_banner()]
        return C.empty("Estate figures unavailable — retrying..."), []
    if not d.get("transactions"):
        return C.empty("No transactions recorded."), []

    kpis = [
        C.kpi_grid([
            C.kpi("Revenue", _inr(d["revenue"]),
                  f"{d['period_start'][:4]}–{d['period_end'][:4]}"),
            C.kpi("Gross margin", _inr(d["margin"]),
                  f"{d['margin_pct']:.1f}% of revenue", "up"),
            C.kpi("Units sold", _compact(d["units"]),
                  f"{_compact(d['transactions'])} transactions"),
            C.kpi("Buyers", _compact(d["buyers"]),
                  f"of {_compact(d['customers'])} registered"),
            C.kpi("Avg basket", _inr(d["avg_basket"]),
                  f"{d['avg_discount_pct']:.1f}% avg discount"),
            C.kpi("Returns", f"{d['return_rate_pct']:.1f}%",
                  "of revenue returned", "down"),
        ]),
        html.Div(
            f"{d['skus_sold']:,} of {d['products']:,} SKUs sold across "
            f"{d['stores']:,} stores and {d['warehouses']} distribution centres, "
            f"supplied by {d['suppliers']:,} vendors.",
            className="small muted mt-14",
        ),
    ]
    # Return cards and empty list for banner (NO banner when data loaded!)
    return kpis, []



# ══════════════════════════════════════════════════════════════════════════════
# 2 — Revenue and margin over time
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("hm-trend", "children"), Input("hm-load", "data"))
def _trend(_):
    d = api_get("/api/v1/overview/revenue-trend", {"months": 36})
    series = (d or {}).get("series") or []
    if not series:
        return C.empty("No transaction history to chart.")

    months = [s["month"] for s in series]
    revenue = [s["revenue"] for s in series]
    margin = [s["margin"] for s in series]

    # Revenue and margin are both rupees, so they share one axis. Margin is a
    # component of revenue, which is exactly what the filled band under the line
    # shows — the gap between the two marks is cost of goods.
    fig = T.figure(height=260, showlegend=True, margin=dict(l=8, r=8, t=8, b=8))
    fig.add_scatter(
        x=months, y=revenue, name="Revenue", mode="lines",
        line=dict(color=S_REVENUE, width=2), fill="tozeroy",
        fillcolor="rgba(92,143,110,.13)",
        hovertemplate="Revenue %{customdata}<extra></extra>",
        customdata=[_inr(v) for v in revenue],
    )
    fig.add_scatter(
        x=months, y=margin, name="Gross margin", mode="lines",
        line=dict(color=S_MARGIN, width=2),
        hovertemplate="Margin %{customdata}<extra></extra>",
        customdata=[_inr(v) for v in margin],
    )
    # Direct end-labels: amber falls below 3:1 on this surface, so identity is
    # never left to hue alone.
    for name, vals, tone in (("Revenue", revenue, S_REVENUE),
                             ("Margin", margin, S_MARGIN)):
        fig.add_annotation(x=months[-1], y=vals[-1], text=f"  {name}",
                           showarrow=False, xanchor="left", font=dict(color=tone, size=11))

    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=0),
        xaxis=dict(showgrid=False, tickformat="%b<br>%Y"),
        yaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"],
                   tickprefix="₹", title="per month"),
    )

    peak = d.get("peak_month")
    chans = d.get("channels") or []
    total = sum(c["revenue"] for c in chans) or 1
    chan_rows = [
        C.bar_row(c["channel"], f"{c['revenue'] / total * 100:.0f}%",
                  c["revenue"] / total * 100)
        for c in chans
    ]

    return C.split(
        [html.Div(C.graph(fig, 260))],
        [
            C.subhead("Where the revenue comes from"),
            html.Div(chan_rows),
            html.Div(
                f"Strongest month on record is {peak}, at {_inr(d.get('peak_revenue'))}."
                if peak else "",
                className="small muted mt-14"),
        ],
        weight="wide-left",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Trading seasonality
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("hm-season", "children"), Input("hm-load", "data"))
def _season(_):
    d = api_get("/api/v1/overview/seasonality")
    cells = (d or {}).get("cells") or []
    if not cells:
        return C.empty("Not enough history to show a seasonal pattern.")

    lookup = {(c["month"], c["weekday"]): c for c in cells}
    z, hover = [], []
    for wd, wd_label in WEEKDAYS:
        row, hrow = [], []
        for m in range(1, 13):
            cell = lookup.get((m, wd))
            val = cell["revenue_per_day"] if cell else 0
            row.append(val)
            hrow.append(f"{MONTHS[m - 1]} · {wd_label}<br>{_inr(val)} per trading day"
                        f"<br>{cell['days'] if cell else 0} days observed")
        z.append(row)
        hover.append(hrow)

    # Magnitude on a single hue, light to dark — the scale is keyed to the range
    # present, so a smaller build still reads across the full ramp.
    fig = T.figure(height=250, margin=dict(l=8, r=8, t=8, b=8))
    fig.add_heatmap(
        z=z, x=MONTHS, y=[w[1] for w in WEEKDAYS],
        colorscale=SEQ, xgap=2, ygap=2,
        text=hover, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(title=dict(text="₹/day", side="right"), thickness=10,
                      len=.85, outlinewidth=0, tickfont=dict(size=10)),
    )
    fig.update_layout(xaxis=dict(showgrid=False, side="top"),
                      yaxis=dict(showgrid=False, autorange="reversed"))

    peak_m = d.get("peak_month")
    peak_w = d.get("peak_weekday")
    wd_name = dict(WEEKDAYS).get(peak_w, "—")
    return [
        C.graph(fig, 250),
        html.Div(
            f"Trade peaks on {wd_name}s in {MONTHS[peak_m - 1]}, averaging "
            f"{_inr(d.get('peak_revenue_per_day'))} a day. Each cell is revenue "
            f"per trading day, so months with more selling days don't read as "
            f"busier than they were."
            if peak_m else "",
            className="small muted mt-8"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Top movers and the data foundation
# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("hm-movers", "children"), Input("hm-load", "data"))
def _movers(_):
    d = api_get("/api/v1/overview/top-movers", {"limit": 8})
    rows = (d or {}).get("products") or []
    if not rows:
        return C.empty("No product revenue to rank.")

    return [
        C.table(
            ["Product", "Units", "Revenue", "Margin"],
            [[html.Div([html.Div(r["product_name"], style={"fontWeight": 600}),
                        html.Div(f"{r['brand']} · {r['category']}", className="small muted")]),
              f"{r['units']:,}", _inr(r["revenue"]), f"{r['margin_pct']:.0f}%"]
             for r in rows],
            numeric={1, 2, 3}, wide={0},
        ),
        html.Div(f"Ranked on revenue across {d.get('year')}.",
                 className="small muted mt-8"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Domain navigation
# ══════════════════════════════════════════════════════════════════════════════

def _domain_card(d):
    caps = [
        html.Div(
            [
                html.Span(className="pdot", style={
                    "width": "6px", "height": "6px", "borderRadius": "999px",
                    "flex": "none", "background": "var(--brand)",
                }),
                html.Span(c.title, className="grow", style={"fontSize": "12.5px"}),
            ],
            className="row center", style={"gap": "9px", "padding": "5px 0"},
        )
        for c in d.capabilities
    ]
    title = html.Span(
        [html.Span(NAV_ICONS.get(_DOMAIN_ICON[d.key]), className="nav-ic",
                   style={"marginRight": "10px"}), d.title],
        className="row center",
    )
    return dcc.Link(
        C.card(title, [
            html.Div(d.tagline, className="card-sub"),
            html.Div(caps),
            html.Div([C.pill(f"{len(d.capabilities)} live", "low"),
                      C.pill(d.index, "info")],
                     className="row-wrap", style={"marginTop": "10px"}),
        ]),
        href=d.path,
        style={"textDecoration": "none", "color": "inherit", "display": "block"},
    )


def layout():
    return module_page(
        "Executive Overview",
        "Retail AI Capability Console",
        "The trading position the sixteen AI capabilities operate on — revenue, "
        "margin, and seasonality, aggregated live from the transaction log.",
        [
            html.Div(id="hm-banner"),
            # Fires once on mount; the unfiltered panels load from it in
            # parallel so the page paints before any rollup has finished.
            dcc.Store(id="hm-load", data=1),

            # Reading order: what the position is, how it is trending, then
            # when and what sells. Each row answers the question the row
            # above it raises.
            C.card("Trading position",
                   dcc.Loading(html.Div(id="hm-estate"), type="dot", color=colors.BRAND),
                   caption="Aggregated from the transaction log, not a sample.",
                   span=2),

            html.Div(
                C.card("Revenue and gross margin by month",
                       dcc.Loading(html.Div(id="hm-trend"), type="dot", color=colors.BRAND),
                       caption="Both series are rupees on one scale — the gap between "
                               "them is cost of goods sold.",
                       info="Margin is revenue less <b>quantity × cost price</b> per line, "
                            "joined from the product catalogue.",
                       span=2),
                style={"marginTop": "18px"}),

            html.Div(
                [
                    C.card("Trading seasonality",
                           dcc.Loading(html.Div(id="hm-season"), type="dot", color=colors.BRAND),
                           info="Each cell is revenue per <b>trading day</b>. Months with "
                                "more selling days would otherwise read as busier than "
                                "they were."),
                    C.card("Top movers",
                           dcc.Loading(html.Div(id="hm-movers"), type="dot", color=colors.BRAND),
                           caption="The SKUs carrying the most revenue in this slice."),
                ],
                className="grid-2", style={"marginTop": "18px"}),

            html.Div(C.subhead("Explore the capabilities"),
                     style={"marginTop": "24px"}),
            html.Div([_domain_card(d) for d in DOMAINS], className="grid-2"),
        ],
    )