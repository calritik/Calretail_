"""
Domain 03 — Operational Efficiency (notebooks 09–12).

Four capability cards — all data-driven, served by FastAPI:
  1. Smart Inventory Management    (nb 09)
  2. Automated Replenishment       (nb 10)
  3. Warehouse Optimization        (nb 11)
  4. Logistics & Route Opt.        (nb 12)
"""

from __future__ import annotations

import math

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from frontend_dash.components import cards as C
from frontend_dash.components.layout import module_page
from frontend_dash.services.api import api_get, api_post, last_failure
from frontend_dash.services.capabilities import OPERATIONS as D
from frontend_dash.services.capabilities import cap
from frontend_dash.theme import chart_theme as T
from frontend_dash.theme import colors

dash.register_page(__name__, path=D.path, name=D.title)

CAP = {c.key: c for c in D.capabilities}

# Risk pill class -> the status colour that encodes the same meaning in charts,
# so a "Critical" bar and a "Critical" pill are never two different oranges.
RISK_TONE = {"high": colors.ACCENT, "medium": colors.WARN,
             "low": colors.OK, "neutral": colors.BRAND3}

# Order lifecycle status -> status colour. Delivered is the healthy end state,
# Cancelled the failed one; everything in between is still in flight and reads
# as brand blue rather than as good or bad news.
STATUS_TONE = {"Delivered": colors.OK, "Shipped": colors.BRAND2,
               "Confirmed": colors.BRAND3, "Placed": colors.BRAND3,
               "Returned": colors.WARN, "Cancelled": colors.ACCENT}


def _inr(v: float) -> str:
    """Indian currency, stepped into Lakh/Crore so KPI tiles never overflow."""
    v = float(v or 0)
    if abs(v) >= 1_00_00_000:
        return f"₹{v / 10000000:.2f}Cr"
    if abs(v) >= 1_00_000:
        return f"₹{v / 100000:.1f}L"
    return f"₹{v:,.0f}"


def _unavailable(path: str, no_data: str):
    """
    Empty state that distinguishes "this endpoint isn't there" from "this
    endpoint returned nothing".

    Both look identical to api_get (it returns None either way), and saying
    "no sales history for this product" when the route actually 404s sends the
    reader off to check the data for a problem that is in the API surface.
    """
    if last_failure(path) == "missing":
        return C.empty(f"This view needs {path}, which the backend does not serve yet.")
    if last_failure(path):
        return C.empty("Backend did not respond to this request.")
    return C.empty(no_data)


def _product_options(limit: int = 60):
    rows = api_get("/api/v1/merchandising/products", {"limit": limit}) or []
    return [{"label": f"{r['product_name']} ({r['category']})",
             "value": r["product_id"]} for r in rows]


def _warehouse_options():
    rows = api_get("/api/v1/operations/warehouses") or []
    return [{"label": f"{r['warehouse_name']} ({r['type']})",
             "value": r["warehouse_id"]} for r in rows]


def _store_options(limit: int = 200):
    # The route caps at 200 and the network is 150 stores, so this is the whole
    # estate — the dropdown is never a truncated view of it.
    rows = api_get("/api/v1/operations/stores", {"limit": limit}) or []
    return [{"label": f"{r['store_name']} ({r['city']})",
             "value": r["store_id"]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Smart Inventory Management
# ══════════════════════════════════════════════════════════════════════════════

def _card_inventory(prods):
    rows = (api_get("/api/v1/operations/inventory-health") or {}).get("results") or []
    if not rows:
        body = C.empty("Inventory health scores unavailable.")
        return C.card(cap("ops", "inventory").title, body,
                      caption="Per-SKU health scoring across stores and distribution centres.",
                      info="<b>Score:</b> days of cover, stockout probability and supplier "
                           "reliability collapsed into one 0-1 health index.",
                      span=2)

    at_risk = [r for r in rows if colors.risk_class(r.get("risk_label")) == "high"]
    overstocked = [r for r in rows if r.get("overstock_flag")]
    avg_health = sum(float(r.get("health_score") or 0) for r in rows) / len(rows)

    # Split the risk mix by location type — a store stockout and a DC stockout
    # carry very different remedies, so the two are never pooled into one bar.
    labels, locs = [], []
    for r in rows:
        if r.get("risk_label") not in labels:
            labels.append(r.get("risk_label"))
        loc = (r.get("location_type") or "—").title()
        if loc not in locs:
            locs.append(loc)
    labels.sort(key=lambda l: {"high": 0, "medium": 1, "low": 2}.get(colors.risk_class(l), 3))

    fig = T.figure(height=190, showlegend=True, margin=dict(l=8, r=8, t=4, b=4))
    for lab in labels:
        fig.add_bar(
            x=locs,
            y=[sum(1 for r in rows
                   if r.get("risk_label") == lab
                   and (r.get("location_type") or "—").title() == loc) for loc in locs],
            name=lab, width=.42,
            marker_color=RISK_TONE.get(colors.risk_class(lab), colors.BRAND3),
            hovertemplate="%{x} · " + str(lab) + "<br>%{y} SKUs<extra></extra>",
        )
    fig.update_layout(barmode="stack",
                      yaxis=dict(title="SKUs", showgrid=True, gridcolor=colors.LIGHT["grid"]))

    worst = sorted(rows, key=lambda r: (float(r.get("health_score") or 0),
                                        -float(r.get("stockout_risk") or 0)))[:8]
    # Five columns, not seven. Seven each demanded their header's longest word
    # ("LOCATION", "STOCKOUT", "HEALTH"…) and together they outgrew the half-card
    # this table sits in, forcing a sideways scroll to read a row. Two pairs
    # collapse without losing anything: where a SKU sits belongs with the SKU,
    # and a health score and its risk label are one judgement, not two.
    tbl = []
    for r in worst:
        where = r.get("location_name") or r.get("store_id") or r.get("warehouse_id") or "—"
        loc_type = r.get("location_type", "")
        tbl.append([
            html.Div([
                html.Div(r.get("product_name", "—"), style={"fontWeight": 600}),
                html.Div(f"{r.get('category', '')} · {where}".strip(" ·"),
                         className="small muted"),
                html.Div(loc_type, className="small muted") if loc_type else None,
            ]),
            f"{r.get('stock_qty', 0):,}",
            f"{float(r.get('days_cover') or 0):,.0f}",
            f"{float(r.get('stockout_risk') or 0) * 100:.0f}%",
            html.Div([
                html.Div(f"{float(r.get('health_score') or 0):.2f}",
                         className="tabular", style={"fontWeight": 700}),
                C.pill(r.get("risk_label", "—"), r.get("risk_label", "")),
            ], className="cell-stack"),
        ])

    # Markdown buy-list: the opposite failure to stockout — SKUs with negligible
    # stockout risk and cover far past the replenishment cycle, where a markdown
    # frees capital. Server-computed and ranked by the value tied up.
    md = api_get("/api/v1/operations/markdown-candidates", {"top_n": 8}) or {}
    # Five columns for the same reason as the buy-list above: six made this
    # table wider than the half-card it lives in. The capital tied up in a SKU
    # and the markdown suggested against it are one recommendation — "this much
    # money is stuck, cut this much off" — so they share a cell.
    md_rows = []
    for c in md.get("candidates") or []:
        md_rows.append([
            html.Div([html.Div(c["product_name"], style={"fontWeight": 600}),
                      html.Div(c["category"], className="small muted")]),
            f"{c['stock']:,}",
            f"{c['days_cover']:,.0f}",
            f"{c['stockout_risk_pct']:.0f}%",
            html.Div([
                html.Div(_inr(c["stock_value"]), className="tabular",
                         style={"fontWeight": 700}),
                C.pill(f"−{c['suggested_markdown_pct']}%", "medium"),
            ], className="cell-stack"),
        ])

    return C.card(
        cap("ops", "inventory").title,
        [
            # Two opposite failures, one per column: too little stock on the
            # left, too much on the right. Stacked they read as one long list
            # and the contrast — which is the whole point of the card — is lost.
            C.split(
                [
                    C.kpi_grid([
                        C.kpi("SKUs monitored", f"{len(rows):,}", "stores + DCs"),
                        C.kpi("At risk", f"{len(at_risk):,}", "stockout exposure", "down"),
                        # Sample-scoped, like the three KPIs beside it. The
                        # estate-wide overstock count lives under the markdown
                        # table instead — mixing the two scopes in one grid read
                        # as "50 SKUs monitored, 10,411 overstocked".
                        C.kpi("Overstocked", f"{len(overstocked):,}",
                              "capital tied up", "down"),
                        C.kpi("Avg health score", f"{avg_health:.2f}", "0 = critical · 1 = healthy"),
                    ]),
                    html.Div(C.graph(fig, 190), className="mt-14"),
                    C.subhead("Weakest positions — the replenishment buy-list"),
                    C.table(["Product", "Stock", "Days cover", "Stockout", "Health"],
                            tbl, numeric={1, 2, 3}, wide={0}),
                ],
                [
                    C.subhead("Markdown candidates — overstocked / idle capital"),
                    C.table(["Product", "Stock", "Days cover", "Stockout",
                             "Tied up · markdown"],
                            md_rows, numeric={1, 2, 3, 4}, wide={0})
                    if md_rows else _unavailable(
                        "/api/v1/operations/markdown-candidates",
                        "No SKU currently carries enough idle cover to mark down."),
                    html.Div(
                        f"{md.get('overstocked_skus', 0):,} positions across the estate hold "
                        f"{_inr(md.get('capital_tied_up', 0))} above their planned level. "
                        f"Clearing the SKUs above at the suggested markdowns recovers "
                        f"~{_inr(md.get('freed_at_markdown', 0))}.",
                        className="small muted mt-8") if md_rows else None,
                    C.subhead("Datewise demand for one SKU"),
                    html.Div(
                        dcc.Dropdown(id="op-inv-prod", options=prods,
                                     value=prods[0]["value"] if prods else None,
                                     clearable=False, className="dash-dropdown grow",
                                     placeholder="Select a product"),
                        className="cp-row",
                    ),
                    html.Div(id="op-inv-datewise"),
                ],
            ),
        ],
        caption="Where stock is thin (buy-list) and where it is idle (markdown list) — plus the "
                "real daily-demand history behind any single SKU.",
        info="<b>Score:</b> days of cover, stockout probability and supplier reliability "
             "collapsed into one 0-1 health index. <b>Markdown candidates</b> are the opposite "
             "failure — cover far beyond the replenishment cycle with no stockout risk; the "
             "suggested markdown deepens with the excess cover. The <b>datewise</b> view is that "
             "SKU's real units sold per day, so a markdown call is checked against the trend.",
        span=2,
    )


@callback(Output("op-inv-datewise", "children"), Input("op-inv-prod", "value"))
def _inventory_datewise(product_id):
    if not product_id:
        return C.empty("Pick a product to see its daily demand.")
    d = api_get("/api/v1/operations/inventory-timeseries", {"product_id": product_id, "days": 120})
    series = (d or {}).get("series") or []
    if not series:
        return _unavailable("/api/v1/operations/inventory-timeseries",
                            "No daily sales history for this product.")

    fig = T.figure(height=200, margin=dict(l=8, r=8, t=6, b=6))
    fig.add_bar(x=[p["date"] for p in series], y=[p["qty"] for p in series],
                marker_color=colors.BRAND2, width=1.0 * 86400000,
                hovertemplate="%{x|%d %b %Y}<br>%{y} units<extra></extra>")
    # A rolling mean line rides the bars so the trend reads through the daily noise.
    win = 7
    qtys = [p["qty"] for p in series]
    roll = [round(sum(qtys[max(0, i - win + 1):i + 1]) / min(i + 1, win), 2) for i in range(len(qtys))]
    fig.add_scatter(x=[p["date"] for p in series], y=roll, mode="lines",
                    line=dict(color=colors.BRAND, width=2), name="7-day avg",
                    hovertemplate="7-day avg %{y:.1f}<extra></extra>")
    fig.update_layout(hovermode="x unified", showlegend=False,
                      xaxis=dict(type="date", showgrid=False),
                      yaxis=dict(title="Units/day", showgrid=True, gridcolor=colors.LIGHT["grid"]))

    dc = float(d.get("days_cover") or 0)
    return [
        C.kpi_grid([
            C.kpi("Current stock", f"{d.get('current_stock', 0):,}", "units on hand"),
            C.kpi("Avg daily demand", f"{d.get('avg_daily_qty', 0):.2f}",
                  f"over {d.get('window_days', 0)} days"),
            C.kpi("Days of cover", f"{dc:,.0f}", "at current pace",
                  "down" if dc > 180 else ""),
        ]),
        html.Div(C.graph(fig, 200), className="mt-14"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Automated Replenishment
# ══════════════════════════════════════════════════════════════════════════════

def _card_replenishment(opts):
    return C.card(
        cap("ops", "replenishment").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="op-repl-prod", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a product"),
                    html.Button("Plan order", id="op-repl-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="op-repl-out"),
        ],
        caption="Reorder quantity, timing and cost derived from live stock position and supplier lead time.",
        info="<b>Policy:</b> a min/max rule — when stock falls to the <b>reorder point</b> "
             "(demand over lead time plus safety stock), order back up to max. Supplier "
             "reliability widens the safety buffer.",
        # Full width, like the other three cards on this page. Half-width, this
        # card and Warehouse Optimization differed by ~430px of height and the
        # row read as broken; full width both of them split into two columns and
        # the page becomes four even bands instead.
        span=2,
    )


@callback(Output("op-repl-out", "children"),
          Input("op-repl-go", "n_clicks"), State("op-repl-prod", "value"))
def _replenishment(_n, product_id):
    if not product_id:
        return C.empty("Pick a product to plan a replenishment order.")
    d = api_post("/api/v1/operations/replenishment", {"product_id": product_id})
    if not d:
        return C.empty("No replenishment plan returned for this product.")

    rel = float(d.get("supplier_reliability") or 0)
    urgent = bool(d.get("urgency_flag"))
    stock = float(d.get("current_stock") or 0)

    # Four thresholds on one axis: where stock sits now against the policy bands
    # is the whole decision, and it only reads at a glance side by side.
    bands = [("Safety stock", float(d.get("safety_stock") or 0), colors.ACCENT),
             ("Reorder point", float(d.get("reorder_point") or 0), colors.WARN),
             ("Current stock", stock, colors.BRAND),
             ("Max stock", float(d.get("max_stock") or 0), colors.BRAND3)]
    fig = T.figure(height=170, margin=dict(l=8, r=8, t=4, b=4))
    fig.add_bar(y=[b[0] for b in bands], x=[b[1] for b in bands], orientation="h",
                marker_color=[b[2] for b in bands], width=.6,
                hovertemplate="%{y}<br>%{x:,.0f} units<extra></extra>")
    fig.update_layout(hovermode="closest",
                      xaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"], title="Units"))

    # Left: the order to place. Right: the policy thresholds it came from.
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("Reorder qty", f"{d.get('reorder_qty', 0):,} units"),
                    C.kpi("Lead time", f"{float(d.get('lead_time_days') or 0):.0f} days",
                          d.get("supplier_name", "")),
                    C.kpi("Estimated cost", _inr(d.get("estimated_cost")),
                          "at last landed price"),
                ]),
                html.Div(C.pill("Urgent — order today" if urgent else "Within tolerance",
                                "high" if urgent else "low"), className="mt-14"),
                html.Div(C.graph(fig, 190), className="mt-14"),
            ],
            [
                C.subhead("Policy thresholds"),
                C.stat_list([
                    ("Supplier", d.get("supplier_name", "—")),
                    ("Safety stock", f"{d.get('safety_stock', 0):,} units"),
                    ("Reorder point", f"{d.get('reorder_point', 0):,} units"),
                    ("Max stock", f"{d.get('max_stock', 0):,} units"),
                ]),
                html.Div(C.bar_row("Supplier reliability", f"{rel * 100:.0f}%", rel * 100,
                                   "ok" if rel >= .8 else "warn" if rel >= .6 else "danger"),
                         className="mt-14"),
            ],
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Warehouse Optimization
# ══════════════════════════════════════════════════════════════════════════════

def _card_warehouse(opts):
    return C.card(
        cap("ops", "warehouse").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="op-wh-sel", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a warehouse"),
                    html.Button("Re-slot", id="op-wh-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="op-wh-out"),
        ],
        caption="ABC velocity classing turned into a slotting plan — fastest movers pulled into the golden zone.",
        info="<b>Method:</b> SKUs ranked by movement, cut at the 80/95% cumulative "
             "thresholds into <b>A/B/C</b> classes, then mapped onto pick zones by "
             "walking distance. Savings are modelled travel time, not headcount.",
        # The densest card on the page: KPIs, a donut, a three-row legend and
        # two tables. At half width it ran ~850px; split across the full grid it
        # halves, and the six-column movers table stops being cramped.
        span=2,
    )


@callback(Output("op-wh-out", "children"),
          Input("op-wh-go", "n_clicks"), Input("op-wh-sel", "value"))
def _warehouse(_n, warehouse_id):
    if not warehouse_id:
        return C.empty("Select a warehouse to build a slotting plan.")
    d = api_get("/api/v1/operations/warehouse-optimization", {"warehouse_id": warehouse_id})
    if not d:
        return C.empty("No slotting plan returned for this warehouse.")

    summary = d.get("class_summary") or {}
    # The plan is ~330 SKU rows — far too many to list, and the operator only acts
    # on the class mix plus the handful of movers that dominate pick travel.
    plan = d.get("slotting_plan") or []
    if not summary and not plan:
        return C.empty("This warehouse has no movement history to slot against.")

    order = [k for k in ("A", "B", "C") if k in summary] or list(summary)
    counts = [int(summary.get(k) or 0) for k in order]
    total = sum(counts) or 1

    fig = T.figure(height=200, showlegend=True, margin=dict(l=8, r=8, t=4, b=4))
    fig.add_pie(labels=[f"Class {k}" for k in order], values=counts, hole=.58, sort=False,
                marker=dict(colors=[colors.BRAND, colors.BRAND2, colors.BRAND3][:len(order)],
                            line=dict(width=0)),
                textinfo="percent", textfont=dict(size=11),
                hovertemplate="%{label}<br>%{value} SKUs (%{percent})<extra></extra>")
    fig.update_layout(hovermode="closest")

    # The ABC pill moves into the product cell and "Recommended zone" becomes
    # "Target zone": six columns in a half-card could not fit, and "RECOMMENDED"
    # was on its own the widest header in the console — a single word no column
    # could be narrower than.
    top = sorted(plan, key=lambda r: -float(r.get("total_movements") or 0))[:6]
    tbl = []
    for r in top:
        moved = (r.get("assigned_zone") or "") != (r.get("recommended_zone") or "")
        tbl.append([
            html.Div([
                html.Div(r.get("product_name", "—"), style={"fontWeight": 600}),
                html.Div(r.get("category", ""), className="small muted"),
                C.pill(f"Class {r.get('abc_class', '—')}", "info"),
            ], className="cell-stack"),
            f"{float(r.get('total_movements') or 0):,.0f}",
            f"{float(r.get('avg_daily_demand') or 0):,.1f}",
            html.Div([html.Div(r.get("recommended_zone", "—")),
                      html.Div("re-slot" if moved else "already slotted",
                               className="small muted")]),
            f"{float(r.get('pick_time_savings_pct') or 0):.0f}%",
        ])

    # Plain-language legend so a non-technical reader knows what A/B/C mean and
    # why a class earns its zone — the classing is only useful if it's understood.
    def _legend_row(cls, tone, headline, detail):
        return html.Div(
            [C.pill(f"Class {cls}", tone),
             html.Span([html.B(headline + " "), html.Span(detail, className="muted")],
                       className="grow", style={"fontSize": "12.5px"})],
            className="row center", style={"gap": "10px", "padding": "5px 0"},
        )

    legend = html.Div([
        _legend_row("A", "info", "Fast movers.",
                    "The first ~80% of all pick movement — a small set of SKUs picked "
                    "constantly. Slotted into Zone 1, the golden fast-pick shelves nearest dispatch."),
        _legend_row("B", "info", "Steady movers.",
                    "The next ~15% of movement — mid-frequency SKUs slotted into the middle zones."),
        _legend_row("C", "info", "Slow / long tail.",
                    "The final ~5% of movement across the bulk of the catalogue — rarely picked, "
                    "so slotted into the far zones where walking distance matters least."),
    ], className="mt-14")

    class_opts = [{"label": f"Class {k} · {summary.get(k, 0)} SKUs", "value": k} for k in order]

    # Left: the class mix and what the classes mean. Right: the SKUs themselves.
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("Pick time reduction",
                          f"{float(d.get('estimated_pick_time_reduction_pct') or 0):.1f}%",
                          "vs. current slotting", "up"),
                    C.kpi("SKUs slotted", f"{len(plan):,}"),
                    C.kpi("Class A share", f"{counts[0] / total * 100:.0f}%" if counts else "—",
                          "of the golden zone"),
                ]),
                html.Div(C.graph(fig, 200), className="mt-14"),
                legend,
            ],
            [
                C.subhead("Top movers — biggest pick-travel wins"),
                C.table(["Product", "Movements", "Daily demand", "Target zone", "Pick saving"],
                        tbl, numeric={1, 2, 4}, wide={0}),
                C.subhead("Which products sit in each class"),
                html.Div(
                    dcc.Dropdown(id="op-wh-class", options=class_opts,
                                 value=order[0] if order else None,
                                 clearable=False, className="dash-dropdown grow"),
                    className="cp-row",
                ),
                html.Div(id="op-wh-class-out"),
            ],
        ),
    ]


@callback(Output("op-wh-class-out", "children"),
          Input("op-wh-class", "value"), State("op-wh-sel", "value"))
def _warehouse_class(abc_class, warehouse_id):
    if not abc_class or not warehouse_id:
        return C.empty("Pick a class to list its products.")
    d = api_get("/api/v1/operations/warehouse-optimization", {"warehouse_id": warehouse_id})
    plan = (d or {}).get("slotting_plan") or []
    members = sorted([r for r in plan if r.get("abc_class") == abc_class],
                     key=lambda r: -float(r.get("total_movements") or 0))
    if not members:
        return C.empty("No products in this class.")

    rows = [
        [html.Div([html.Div(r.get("product_name", "—"), style={"fontWeight": 600}),
                   html.Div(r.get("category", ""), className="small muted")]),
         f"{float(r.get('total_movements') or 0):,.0f}",
         f"{float(r.get('avg_daily_demand') or 0):,.1f}",
         r.get("recommended_zone", "—")]
        for r in members
    ]
    return [
        html.Div(f"{len(members):,} SKUs in Class {abc_class}", className="small muted mb-10"),
        C.table(["Product", "Movements", "Daily demand", "Target zone"], rows,
                numeric={1, 2}, wide={0}),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Logistics, Route & Fleet Optimization
# ══════════════════════════════════════════════════════════════════════════════

# Free MapLibre raster styles (no access token). chart_theme.js swaps LIGHT->DARK
# when the theme toggle fires, so the tiles match the rest of the surface.
MAP_STYLE_LIGHT = "carto-positron"
MAP_STYLE_DARK = "carto-darkmatter"

# The route map wants live pan/zoom, unlike the static analytical charts, so it
# opts out of the shared no-interaction GRAPH_CONFIG.
MAP_CONFIG = {"displayModeBar": False, "responsive": True,
              "scrollZoom": True, "doubleClick": "reset"}


def _fit_view(lats, lngs):
    """Centre + zoom that frames every stop with a comfortable margin.

    Longitude is padded a touch harder than latitude because the card is wider
    than it is tall, and the zoom is derived from whichever span dominates so a
    single-metro tour comes in close and a coast-to-coast one pulls right back.
    """
    lat_c = (min(lats) + max(lats)) / 2
    lng_c = (min(lngs) + max(lngs)) / 2
    lat_span = (max(lats) - min(lats)) or 0.35
    lng_span = (max(lngs) - min(lngs)) or 0.35
    span = max(lat_span * 1.55, lng_span * 1.15)
    zoom = 4.0 + math.log2(22.0 / max(span, 0.35))
    return lat_c, lng_c, max(3.3, min(zoom, 9.8))


def _route_map(olat, olng, lats, lngs, path_lat, path_lng, pts, sizes):
    """A real, interactive India map of the solved delivery tour.

    Layered for depth: a soft glow under the route line, a white halo under each
    pin, the numbered stop on top, and the depot rendered as its own accent
    marker. Everything is a Scattermap trace, so pan/zoom stay live.
    """
    lat_c, lng_c, zoom = _fit_view(path_lat, path_lng)

    fig = go.Figure()

    # Route line — a translucent glow beneath a crisp stroke reads as a drawn
    # path rather than a hairline, and survives against busy tiles.
    fig.add_trace(go.Scattermap(
        lat=path_lat, lon=path_lng, mode="lines",
        line=dict(color=colors.BRAND2, width=8),
        opacity=0.22, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(
        lat=path_lat, lon=path_lng, mode="lines",
        line=dict(color=colors.BRAND, width=2.6),
        hoverinfo="skip", showlegend=False))

    # White halo lifts every pin off the map, whichever theme is showing.
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lngs, mode="markers",
        marker=dict(size=[s + 6 for s in sizes], color="#ffffff"),
        opacity=0.95, hoverinfo="skip", showlegend=False))

    # Numbered stops, sized by order load. The sequence number sits inside.
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lngs, mode="markers+text",
        marker=dict(size=sizes, color=colors.BRAND),
        text=[str(p["n"]) for p in pts], textposition="middle center",
        textfont=dict(size=12, color="#ffffff"),
        customdata=[[p["n"], p["city"], p["sid"] or "—", p["items"]] for p in pts],
        hovertemplate="<b>Stop %{customdata[0]} · %{customdata[1]}</b><br>"
                      "%{customdata[2]} · %{customdata[3]} items<extra></extra>",
        showlegend=False))

    # City names as a light label layer, offset above the pins.
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lngs, mode="text",
        text=[p["city"] for p in pts], textposition="top center",
        textfont=dict(size=10.5, color=colors.INK_SOFT),
        hoverinfo="skip", showlegend=False))

    # Dispatch depot — accent glow + solid pin, on its own so it never reads as
    # just another stop.
    fig.add_trace(go.Scattermap(
        lat=[olat], lon=[olng], mode="markers",
        marker=dict(size=38, color=colors.ACCENT),
        opacity=0.20, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scattermap(
        lat=[olat], lon=[olng], mode="markers+text",
        marker=dict(size=22, color=colors.ACCENT),
        text=["◆"], textposition="middle center",
        textfont=dict(size=13, color="#ffffff"),
        hovertemplate="<b>Dispatch depot</b><br>route origin & return<extra></extra>",
        showlegend=False))
    fig.add_trace(go.Scattermap(
        lat=[olat], lon=[olng], mode="text",
        text=["Dispatch DC"], textposition="bottom center",
        textfont=dict(size=10.5, color=colors.ACCENT_INK),
        hoverinfo="skip", showlegend=False))

    fig.update_layout(
        height=440,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=T.FONT),
        hovermode="closest",
        hoverlabel=dict(bgcolor=colors.LIGHT["tooltip_bg"],
                        bordercolor=colors.LIGHT["tooltip_line"],
                        font=dict(family=T.FONT, size=11.5,
                                  color=colors.LIGHT["tooltip_ink"])),
        map=dict(style=MAP_STYLE_LIGHT,
                 center=dict(lat=lat_c, lon=lng_c), zoom=zoom,
                 bearing=0, pitch=0),
        transition=dict(duration=0),
    )
    return fig


def _card_route(opts):
    return C.card(
        cap("ops", "route").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="op-route-wh", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a dispatch origin"),
                    html.Button("Optimise route", id="op-route-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="op-route-out"),
        ],
        caption="The day's drop sequence out of one distribution centre, plotted on a live map "
                "against the distance a naive round trip would cover.",
        info="<b>Solver:</b> a nearest-neighbour tour with 2-opt improvement over the "
             "haversine distance matrix, closing back at the origin. <b>Baseline</b> is the "
             "unordered depot-and-back-again pattern the route replaces. <b>Map:</b> each "
             "pin sits at the outlet's real coordinates on OpenStreetMap/Carto tiles — pan "
             "and scroll to zoom; the numbered pins follow the solved visiting order and are "
             "sized by the orders on board.",
        span=2,
    )


@callback(Output("op-route-out", "children"),
          Input("op-route-go", "n_clicks"), State("op-route-wh", "value"))
def _route(_n, warehouse_id):
    if not warehouse_id:
        return C.empty("Select a warehouse to optimise its delivery route.")
    d = api_post("/api/v1/operations/route-optimization", {"warehouse_id": warehouse_id})
    if not d:
        return C.empty("No route could be solved for this warehouse.")

    stops = d.get("route") or []
    origin = d.get("origin") or {}
    if not stops or origin.get("lat") is None:
        return C.empty("This warehouse has no open deliveries to route.")

    seq = d.get("route_order") or []
    stop_ids = seq[1:-1] if len(seq) >= len(stops) + 2 else [""] * len(stops)

    # Several stops can share a city centroid; on a real map a big nudge would
    # fling a marker into the next district, so we fan co-located stops out in a
    # tight spiral (~4-6 km) — enough to unstack the pins without lying about
    # where the outlet actually is.
    seen: dict[tuple, int] = {}
    pts = []
    for i, s in enumerate(stops):
        key = (round(float(s.get("lat") or 0), 3), round(float(s.get("lng") or 0), 3))
        k = seen.get(key, 0)
        seen[key] = k + 1
        ang = k * 2.399963  # golden angle — successive pins never line up
        pts.append({
            "lat": float(s.get("lat") or 0) + .05 * k * math.cos(ang),
            "lng": float(s.get("lng") or 0) + .05 * k * math.sin(ang),
            "city": s.get("city", "—"),
            "items": int(s.get("items") or 0),
            "sid": stop_ids[i] if i < len(stop_ids) else "",
            "n": i + 1,
        })

    olat, olng = float(origin["lat"]), float(origin["lng"])
    lats = [p["lat"] for p in pts]
    lngs = [p["lng"] for p in pts]
    # The tour is drawn on real tiles now, so the line follows true coordinates:
    # depot -> each drop in solved order -> back to the depot.
    path_lat = [olat] + lats + [olat]
    path_lng = [olng] + lngs + [olng]

    items = [p["items"] for p in pts]
    lo, span = min(items), (max(items) - min(items)) or 1
    # Marker area scales with load; floor of 17 keeps the sequence number legible
    # inside the smallest pin.
    sizes = [17 + (v - lo) / span * 21 for v in items]

    fig = _route_map(olat, olng, lats, lngs, path_lat, path_lng, pts, sizes)

    saving = float(d.get("saving_pct") or 0)
    crumbs = []
    for i, label in enumerate(seq):
        if i:
            crumbs.append(html.Span("›", className="small muted"))
        crumbs.append(html.Span(label, className="chip"))

    # The map is the card. Everything numeric about the tour moves into a
    # narrower left column so the map keeps roughly 60% of the width instead of
    # being a 440px band with four separate blocks stacked underneath it.
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("Optimised distance",
                          f"{float(d.get('optimised_distance_km') or 0):,.0f} km"),
                    C.kpi("Baseline distance",
                          f"{float(d.get('baseline_distance_km') or 0):,.0f} km",
                          "unoptimised round trips"),
                    C.kpi("Distance saved", f"{float(d.get('distance_saved_km') or 0):,.0f} km",
                          f"{saving:.1f}% shorter", "up"),
                    C.kpi("Drive time", f"{float(d.get('estimated_time_hrs') or 0):,.1f} hrs",
                          "single vehicle"),
                    C.kpi("Orders on board", f"{d.get('total_orders', 0):,}",
                          f"{len(pts)} drops"),
                ]),
                html.Div(C.bar_row("Distance saved vs. baseline", f"{saving:.1f}%", saving,
                                   "ok" if saving >= 20 else "warn" if saving >= 8
                                   else "danger"),
                         className="mt-14"),
                C.subhead("Visiting order"),
                html.Div(crumbs, className="row-wrap"),
            ],
            dcc.Graph(figure=fig, config=MAP_CONFIG,
                      className="route-map", style={"height": "460px"}),
            weight="wide-right",
        ),
    ]



# ══════════════════════════════════════════════════════════════════════════════


def layout():
    prods = _product_options()
    whs = _warehouse_options()
    banner = [] if (prods or whs) else [C.offline_banner()]
    return module_page(
        D.index, D.title, D.summary,
        banner + [
            html.Div(
                [
                    _card_inventory(prods),
                    _card_replenishment(prods),
                    _card_warehouse(whs),
                    _card_route(whs),
                ],
                className="grid-2",
            )
        ],
    )
