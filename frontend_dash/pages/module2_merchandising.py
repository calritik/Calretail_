"""
Domain 02 — Merchandising (Calsoft Retail AI deck, slide 5).

Five capability cards, every one served by FastAPI.
Structure, idioms and callback shapes follow module1_customer_experience.py.

Performance fix: layout() no longer makes ANY blocking API calls.
All heavy data-fetching is deferred into Dash callbacks so the page shell
renders instantly (~0 ms) and data lazy-loads with Loading spinners.
"""
from __future__ import annotations

import dash
from dash import Input, Output, State, callback, dcc, html

from frontend_dash.components import cards as C
from frontend_dash.components.layout import module_page
from frontend_dash.services.api import api_get, api_post
from frontend_dash.services.capabilities import MERCHANDISING as D
from frontend_dash.services.capabilities import cap
from frontend_dash.theme import chart_theme as T
from frontend_dash.theme import colors

dash.register_page(__name__, path=D.path, name=D.title)

CAP = {c.key: c for c in D.capabilities}


def _inr(v) -> str:
    """Rupees, compacted to Lakh/Crore — an Indian merchandiser reads ₹1.2Cr, not ₹12,000,000."""
    v = float(v or 0)
    if abs(v) >= 1e7:
        return f"₹{v / 10000000:.2f}Cr"
    if abs(v) >= 1e5:
        return f"₹{v / 100000:.1f}L"
    return f"₹{v:,.0f}"


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Dynamic Pricing Engines
# ══════════════════════════════════════════════════════════════════════════════

def _card_pricing():
    return C.card(
        "Dynamic Pricing Engines",
        [
            # Dropdown is populated by callback, not layout()
            dcc.Loading(
                dcc.Dropdown(id="mc-price-prod", options=[], value=None,
                             clearable=False, className="dash-dropdown grow",
                             placeholder="Loading products…"),
                type="circle", color=colors.BRAND,
            ),
            html.Div(
                html.Button("Reprice", id="mc-price-go", className="cp-go", n_clicks=0),
                className="cp-row", style={"marginTop": "8px"},
            ),
            dcc.Loading(html.Div(id="mc-price-out"), type="dot", color=colors.BRAND),
        ],
        caption="One SKU repriced against its competitor set, its stock position and a hard margin floor.",
        info="<b>Model:</b> elasticity-weighted price search bounded by the <b>floor price</b> "
             "(cost + minimum margin). Competitor mean and stock cover pull the recommendation "
             "up or down; the engine never returns a price below the floor.",
        span=2,
    )


@callback(Output("mc-price-prod", "options"),
          Output("mc-price-prod", "value"),
          Input("mc-page-load", "data"))
def _load_product_opts(_):
    """Populate product dropdown lazily after page mount."""
    rows = api_get("/api/v1/merchandising/products", {"limit": 60}) or []
    opts = [{"label": f"{r['product_name']} (₹{r['price']:,.0f})",
             "value": r["product_id"]} for r in rows]
    return opts, (opts[0]["value"] if opts else None)


@callback(Output("mc-price-out", "children"),
          Input("mc-price-go", "n_clicks"), Input("mc-price-prod", "value"))
def _pricing(_n, product_id):
    # The product is an Input, not State, so the card fills in as soon as the
    # dropdown finishes loading. As State it only ever fired once — before the
    # options arrived — and then sat on "Pick a product" with a product already
    # picked, until someone happened to press Reprice. Forecasting and Promotion
    # on this page already work this way.
    if not product_id:
        return C.empty("Pick a product to run the pricing engine.")
    d = api_post("/api/v1/merchandising/dynamic-pricing", {"product_id": product_id})
    if not d:
        return C.empty("No price recommendation returned for this product.")

    delta = float(d.get("price_delta_pct", 0) or 0)
    lift = float(d.get("expected_revenue_lift_pct", 0) or 0)

    labels = ["Floor", "Current", "Recommended", "Avg competitor", "Min competitor"]
    values = [d.get("floor_price", 0), d.get("current_price", 0), d.get("recommended_price", 0),
              d.get("avg_competitor_price", 0), d.get("min_competitor_price", 0)]
    tones = ["#d3d8c4", colors.BRAND2, colors.BRAND,
             colors.CATEGORICAL[3], colors.CATEGORICAL[4]]

    fig = T.figure(height=200, margin=dict(l=8, r=8, t=4, b=4))
    fig.add_bar(y=labels[::-1], x=[float(v or 0) for v in values][::-1],
                orientation="h", marker_color=tones[::-1], width=.6,
                hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>")
    fig.update_layout(hovermode="closest",
                      xaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"], tickprefix="₹"))

    rec = float(d.get("recommended_price", 0) or 0)
    ctx = {
        "current": float(d.get("current_price", 0) or 0),
        "cost": float(d.get("cost_price", 0) or 0),
        "floor": float(d.get("floor_price", 0) or 0),
        "elasticity": float(d.get("price_elasticity", -1.0) or -1.0),
        "comp_avg": float(d.get("avg_competitor_price", 0) or 0),
        "comp_min": float(d.get("min_competitor_price", 0) or 0),
        "recommended": rec,
    }
    # Left: what the engine recommends. Right: what happens if you disagree.
    # Two independent readings, so they sit side by side rather than the second
    # being buried a screen below the first.
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("Current price", f"₹{ctx['current']:,.0f}"),
                    C.kpi("Recommended price", f"₹{rec:,.0f}",
                          f"{delta:+.2f}% vs. current", "up" if delta >= 0 else "down"),
                    C.kpi("Expected revenue lift", f"{lift:+.1f}%", "at the recommended price",
                          "up" if lift >= 0 else "down"),
                    C.kpi("Stock on hand", f"{int(d.get('stock_level', 0) or 0):,}", "units"),
                ]),
                html.Div(C.graph(fig, 200), className="mt-14"),
                html.Div(d.get("rationale", ""), className="small muted mt-8"),
            ],
            [
                html.Div("Override the price and check the numbers",
                         className="card-sub mt-0"),
                html.Div(
                    [
                        html.Span("₹", className="muted",
                                  style={"fontWeight": 700, "fontSize": "15px"}),
                        # Whole rupees, and the seeded value is rounded to match.
                        # step=1 with a value of 2584.4 failed the field's own
                        # step constraint, so the box opened :invalid and drew
                        # its digits in red before anyone had typed anything.
                        # Every other price on this card is shown to the rupee,
                        # so rounding here is what the reader already expects.
                        #
                        # Width fits a five-figure price *plus* the browser's
                        # spin buttons; at 140px the two competed for the space.
                        dcc.Input(id="mc-price-override", type="number", value=round(rec),
                                  min=0, step=1, debounce=False, className="cp-input",
                                  style={"maxWidth": "168px"}),
                        html.Button("Reset to recommended", id="mc-price-reset",
                                    className="chip", n_clicks=0),
                    ],
                    className="cp-row",
                ),
                html.Div(id="mc-price-sim"),
            ],
        ),
        dcc.Store(id="mc-price-ctx", data=ctx),
    ]


@callback(Output("mc-price-override", "value"),
          Input("mc-price-reset", "n_clicks"), State("mc-price-ctx", "data"),
          prevent_initial_call=True)
def _price_reset(_n, ctx):
    # Rounded the same way the field is seeded, so Reset can't put the input
    # back into the :invalid state that step=1 rejects.
    return round(float((ctx or {}).get("recommended", 0) or 0))


@callback(Output("mc-price-sim", "children"),
          Input("mc-price-override", "value"), State("mc-price-ctx", "data"))
def _price_sim(price, ctx):
    ctx = ctx or {}
    current = float(ctx.get("current", 0) or 0)
    if not price or current <= 0:
        return C.empty("Enter a price to project margin and revenue.")
    p = float(price)
    cost = float(ctx.get("cost", 0) or 0)
    floor = float(ctx.get("floor", 0) or 0)
    elasticity = float(ctx.get("elasticity", -1.0) or -1.0)
    comp_avg = float(ctx.get("comp_avg", 0) or 0)
    comp_min = float(ctx.get("comp_min", 0) or 0)

    delta = (p - current) / current * 100
    volume = elasticity * delta
    revenue = delta + volume + (delta * volume / 100)
    margin = (p - cost) / p * 100 if p > 0 else 0
    margin_now = (current - cost) / current * 100 if current > 0 else 0

    flags = []
    if p < cost:
        flags.append(C.pill("below cost — loss-making", "high"))
    elif p < floor:
        flags.append(C.pill(f"below margin floor ₹{floor:,.0f}", "high"))
    if comp_avg:
        if p <= comp_min:
            flags.append(C.pill("undercuts every competitor", "low"))
        elif p <= comp_avg:
            flags.append(C.pill("at or under market average", "low"))
        else:
            flags.append(C.pill(f"{(p - comp_avg) / comp_avg * 100:+.1f}% vs market avg", "medium"))

    return [
        C.kpi_grid([
            C.kpi("Your price", f"₹{p:,.0f}", f"{delta:+.1f}% vs current",
                  "up" if delta >= 0 else "down"),
            C.kpi("Gross margin", f"{margin:.1f}%", f"was {margin_now:.1f}%",
                  "up" if margin >= margin_now else "down"),
            C.kpi("Projected revenue", f"{revenue:+.1f}%", "price × volume effect",
                  "up" if revenue >= 0 else "down"),
            C.kpi("Volume impact", f"{volume:+.1f}%", f"elasticity {elasticity:.2f}",
                  "up" if volume >= 0 else "down"),
        ]),
        html.Div(flags, className="row-wrap mt-14") if flags else None,
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Competitor Price Monitoring
# ══════════════════════════════════════════════════════════════════════════════

def _card_competitor():
    return C.card(
        "Competitor Price Monitoring",
        [
            dcc.Loading(
                # These filters rest at "no filter", so their placeholder is
                # their resting state, not "Loading…" — with value=None the
                # placeholder never clears, and the card sat there claiming to
                # still be loading long after the table beneath it had filled in.
                dcc.Dropdown(id="mc-comp-cat", options=[], value=None,
                             clearable=True, className="dash-dropdown grow",
                             placeholder="All categories"),
                type="circle", color=colors.BRAND,
            ),
            dcc.Loading(html.Div(id="mc-comp-out"), type="dot", color=colors.BRAND),
        ],
        caption="The whole catalogue is swept every run; only the SKUs that breached their price band are listed.",
        info="<b>Method:</b> each SKU's price gap against the competitor mean is turned into a "
             "<b>z-score</b> across the category. Anything beyond the band raises "
             "<b>alert_flag</b> and gets a recommended action.",
    )


def _build_competitor_content(rows):
    """
    The KPI + breach-table view of a competitor sweep.

    Shared by the initial load and the category filter, which previously carried
    two copies of this body — so a column-width or wording fix had to be made
    twice to actually reach the reader.
    """
    total = len(rows)
    alerts = [r for r in rows if r.get("alert_flag")]
    above = sum(1 for r in rows if float(r.get("price_gap_pct", 0) or 0) > 0)
    below = total - above

    worst = sorted(alerts, key=lambda r: -abs(float(r.get("price_gap_pct", 0) or 0)))[:8]
    body = []
    for r in worst:
        gap = float(r.get("price_gap_pct", 0) or 0)
        body.append([
            html.Div([html.Div(r.get("product_name", "—"), style={"fontWeight": 600}),
                      html.Div(r.get("recommended_action", ""), className="small muted")]),
            f"₹{float(r.get('our_price', 0) or 0):,.0f}",
            f"₹{float(r.get('avg_competitor_price', 0) or 0):,.0f}",
            f"{gap:+.1f}%",
            C.pill(r["status"], r["status"]),
        ])

    return [
        C.kpi_grid([
            C.kpi("SKUs monitored", f"{total:,}"),
            C.kpi("Price alerts", f"{len(alerts):,}",
                  f"{len(alerts) / total * 100:.1f}% of the sweep", "down"),
            C.kpi("Above market", f"{above / total * 100:.1f}%", "our price > competitor mean"),
            C.kpi("Below market", f"{below / total * 100:.1f}%", "our price < competitor mean"),
        ]),
        # The product column carries a name plus its recommended action; the four
        # money/status columns only need their own width, so they're pinned
        # narrow and the remainder goes to the text that actually varies.
        html.Div(C.table(["Product", "Our price", "Market avg", "Gap", "Status"],
                         body, numeric={1, 2, 3}, wide={0}, narrow={4}),
                 className="mt-14"),
    ]


@callback(Output("mc-comp-cat", "options"),
          Output("mc-comp-out", "children"),
          Input("mc-page-load", "data"))
def _load_competitor(_):
    """Load competitor data lazily — this triggers the slow notebook."""
    rows = (api_get("/api/v1/merchandising/competitor-monitoring") or {}).get("results") or []
    cat_opts = [{"label": c, "value": c}
                for c in sorted({r["category"] for r in rows if r.get("category")})]

    if not rows:
        return cat_opts, C.empty("Competitor monitoring feed unavailable.")
    return cat_opts, _build_competitor_content(rows)


@callback(Output("mc-comp-out", "children", allow_duplicate=True),
          Input("mc-comp-cat", "value"),
          prevent_initial_call=True)
def _competitor_filter(category):
    rows = (api_get("/api/v1/merchandising/competitor-monitoring") or {}).get("results") or []
    if not rows:
        return C.empty("Competitor monitoring feed unavailable.")
    if category:
        rows = [r for r in rows if r.get("category") == category]
    if not rows:
        return C.empty(f"No monitored SKUs in {category}.")
    return _build_competitor_content(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Promotion Optimization
# ══════════════════════════════════════════════════════════════════════════════

def _card_promotion():
    return C.card(
        "Promotion Optimization",
        [
            dcc.Loading(
                dcc.Dropdown(id="mc-promo-sel", options=[], value=None,
                             clearable=False, className="dash-dropdown grow",
                             placeholder="Loading promotions…"),
                type="circle", color=colors.BRAND,
            ),
            html.Div(
                html.Button("Evaluate", id="mc-promo-go", className="cp-go", n_clicks=0),
                className="cp-row", style={"marginTop": "8px"},
            ),
            dcc.Loading(html.Div(id="mc-promo-out"), type="dot", color=colors.BRAND),
        ],
        caption="Treated versus control revenue for one promotion — the incremental lift net of what the discount ate.",
        info="<b>Method:</b> a matched control cohort is compared against the treated cohort. "
             "<b>Cannibalization</b> is the share of the lift stolen from full-price sales; "
             "above 30% the promotion is moving margin, not volume.",
    )


@callback(Output("mc-promo-sel", "options"),
          Output("mc-promo-sel", "value"),
          Input("mc-page-load", "data"))
def _load_promo_opts(_):
    """Populate promotions dropdown lazily after page mount."""
    rows = api_get("/api/v1/merchandising/promotions") or []
    opts = [{"label": f"{r['promo_type']} ({r['discount_pct'] * 100:.0f}% off) — {r['target_segment']}",
             "value": r["promo_id"]} for r in rows]
    return opts, (opts[0]["value"] if opts else None)


@callback(Output("mc-promo-out", "children"),
          Input("mc-promo-go", "n_clicks"), Input("mc-promo-sel", "value"))
def _promotion(_n, promo_id):
    if not promo_id:
        return C.empty("Select a promotion to evaluate.")
    d = api_get("/api/v1/merchandising/promotion-optimization", {"promo_id": promo_id})
    if not d:
        return C.empty("No experiment result available for this promotion.")

    control = float(d.get("control_revenue", 0) or 0)
    treated = float(d.get("treated_revenue", 0) or 0)
    uplift = float(d.get("uplift_pct", 0) or 0)
    conf = float(d.get("confidence", 0) or 0)
    cann = float(d.get("cannibalization_rate", 0) or 0)
    cell = d.get("product_name") or d.get("product_id", "—")

    fig = T.figure(height=190, showlegend=True, margin=dict(l=8, r=8, t=4, b=4))
    fig.add_bar(x=[cell], y=[control], name="Control", marker_color="#d3d8c4", width=.28,
                hovertemplate="Control ₹%{y:,.0f}<extra></extra>")
    fig.add_bar(x=[cell], y=[treated], name="Treated", marker_color=colors.BRAND, width=.28,
                hovertemplate="Treated ₹%{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="group", hovermode="closest",
                      yaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"], tickprefix="₹"))

    return [
        C.kpi_grid([
            C.kpi("Control revenue", _inr(control)),
            C.kpi("Treated revenue", _inr(treated), f"{uplift:+.1f}% uplift",
                  "up" if uplift >= 0 else "down"),
            C.kpi("Incremental", _inr(d.get("incremental_revenue", 0)), "net of control", "up"),
        ]),
        html.Div(C.graph(fig, 190), className="mt-14"),
        html.Div(
            [
                C.bar_row("Confidence", f"{conf * 100:.0f}%", conf * 100,
                          "ok" if conf >= .8 else "warn" if conf >= .5 else "danger"),
                C.bar_row("Cannibalization", f"{cann * 100:.0f}%", cann * 100,
                          "danger" if cann > .3 else "warn" if cann > .15 else "ok"),
            ],
            className="mt-14",
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Assortment Planning
# ══════════════════════════════════════════════════════════════════════════════

def _card_assortment():
    return C.card(
        "Assortment Planning",
        [
            dcc.Loading(
                dcc.Dropdown(id="mc-assort-region", options=[], value=None,
                             clearable=True, className="dash-dropdown grow",
                             placeholder="All regions"),
                type="circle", color=colors.BRAND,
            ),
            dcc.Loading(html.Div(id="mc-assort-out"), type="dot", color=colors.BRAND),
        ],
        caption="Every SKU scored on its share of its own region's revenue, then read against the regions that already sell it.",
        info="<b>Method:</b> orders joined to products and to each store's region; Cancelled and "
             "Returned orders are excluded because neither leaves revenue on the books. Because "
             "regions differ in size, a SKU is judged on its <b>share of its own region's "
             "revenue</b>, never on rupees. <b>80/20</b> is a real Pareto — the share of selling "
             "SKUs that earn the first 80% of the revenue. An <b>add</b> is proven in at least two "
             "other regions yet earning under half that peer share here; a <b>drop</b> sits below "
             "a quarter of the region's median SKU revenue, with inventory joined so lines that "
             "are stocked but static surface too.",
        # Five KPIs, a chart and two tables: the tallest card on the page by far.
        # Spanning the grid lets the body split into two columns instead of
        # running to roughly three screens.
        span=2,
    )


@callback(Output("mc-assort-region", "options"),
          Output("mc-assort-out", "children"),
          Input("mc-page-load", "data"))
def _load_assortment(_):
    """Load assortment plan lazily — this is the heaviest call."""
    d = api_get("/api/v1/merchandising/assortment-plan") or {}
    rows = d.get("by_region") or []
    region_opts = [{"label": r["region"], "value": r["region"]} for r in rows]

    if not rows:
        return region_opts, C.empty("Assortment plan unavailable.")

    content = _build_assortment_content(d, rows)
    return region_opts, content


def _build_assortment_content(d, rows):
    fig = T.figure(height=200, showlegend=True, margin=dict(l=8, r=8, t=4, b=4))
    fig.add_bar(x=[r["region"] for r in rows], y=[r["add"] for r in rows],
                name="Add", marker_color=colors.BRAND, width=.34,
                hovertemplate="%{x}<br>Add %{y} SKUs<extra></extra>")
    fig.add_bar(x=[r["region"] for r in rows], y=[r["drop"] for r in rows],
                name="Drop", marker_color="#d3d8c4", width=.34,
                hovertemplate="%{x}<br>Drop %{y} SKUs<extra></extra>")
    fig.update_layout(barmode="group",
                      yaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"]))

    region_rows = []
    for r in rows:
        p = float(r.get("pareto_sku_pct", 0) or 0)
        lvl = "high" if p < 43 else "medium" if p < 46 else "low"
        region_rows.append([
            r["region"],
            f"{r['skus_selling']:,}",
            C.money(r["revenue"]),
            C.pill(f"{p:.1f}%", lvl),
            f"+{r['add']:,}",
            f"−{r['drop']:,}",
        ])

    moves = []
    for a in (d.get("add_candidates") or [])[:5]:
        moves.append([
            C.pill("Add", "low"),
            html.Div([html.Div(a["product_name"], style={"fontWeight": 600}),
                      html.Div(f"{a['category']} · {a['status']}",
                               className="small muted")]),
            a["region"],
            C.money(a["opportunity"]),
        ])
    for x in (d.get("drop_candidates") or [])[:5]:
        moves.append([
            C.pill("Drop", "high"),
            html.Div([html.Div(x["product_name"], style={"fontWeight": 600}),
                      html.Div(f"{x['category']} · {x['status']}",
                               className="small muted")]),
            x["region"],
            C.money(x["tied_capital"]),
        ])

    # Left column reads the shape of the estate (totals + the add/drop mix per
    # region); right column is the actionable detail (which regions, which SKUs).
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("SKUs selling", f"{d.get('skus_selling', 0):,}",
                          f"of {d.get('catalogue_skus', 0):,} in the catalogue"),
                    C.kpi("Revenue analysed", C.money(d.get("revenue_total", 0)),
                          f"{d.get('orders_analysed', 0):,} store-attributed orders"),
                    C.kpi("80/20 concentration", f"{float(d.get('pareto_sku_pct', 0) or 0):.1f}%",
                          "of selling SKUs earn 80% of revenue"),
                    C.kpi("Add candidates", f"+{d.get('add_candidates_total', 0):,}",
                          f"{C.money(d.get('opportunity_value', 0))} share shortfall", "up"),
                    C.kpi("Drop candidates", f"−{d.get('drop_candidates_total', 0):,}",
                          f"{C.money(d.get('tied_capital', 0))} stock tied up", "down"),
                ]),
                html.Div(C.graph(fig, 200), className="mt-14"),
            ],
            [
                C.subhead("Per-region position"),
                C.table(["Region", "SKUs selling", "Revenue", "80/20", "Add", "Drop"],
                        region_rows, numeric={1, 2, 4, 5}, wide={0}),
                C.subhead("Recommended moves"),
                C.table(["Move", "Product", "Region", "Value"], moves,
                        numeric={3}, wide={1}, narrow={0}),
            ],
        ),
    ]


@callback(Output("mc-assort-out", "children", allow_duplicate=True),
          Input("mc-assort-region", "value"),
          prevent_initial_call=True)
def _assortment_filter(region):
    d = api_get("/api/v1/merchandising/assortment-plan",
                {"region": region} if region else None) or {}
    rows = d.get("by_region") or []
    if not rows:
        return C.empty("Assortment plan unavailable.")
    return _build_assortment_content(d, rows)


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Demand Forecasting
# ══════════════════════════════════════════════════════════════════════════════

def _card_forecast():
    return C.card(
        "Demand Forecasting",
        [
            dcc.Loading(
                dcc.Dropdown(id="mc-fc-prod", options=[], value=None,
                             clearable=False, className="dash-dropdown grow",
                             placeholder="Loading products…"),
                type="circle", color=colors.BRAND,
            ),
            html.Div(
                html.Button("Forecast", id="mc-fc-go", className="cp-go", n_clicks=0),
                className="cp-row", style={"marginTop": "8px"},
            ),
            dcc.Loading(html.Div(id="mc-fc-out"), type="dot", color=colors.BRAND),
        ],
        caption="Observed demand carried forward into a 30-day forecast, with the band the model is willing to commit to.",
        info="<b>Model:</b> a global multi-product regressor over calendar, price and promotion "
             "features. The shaded band is the prediction interval the model is willing to "
             "commit to over the 30-day horizon.",
        span=2,
    )


# Share product opts: mc-fc-prod is wired to the same callback as mc-price-prod
@callback(Output("mc-fc-prod", "options"),
          Output("mc-fc-prod", "value"),
          Input("mc-page-load", "data"))
def _load_fc_product_opts(_):
    rows = api_get("/api/v1/merchandising/products", {"limit": 60}) or []
    opts = [{"label": f"{r['product_name']} (₹{r['price']:,.0f})",
             "value": r["product_id"]} for r in rows]
    return opts, (opts[0]["value"] if opts else None)


@callback(Output("mc-fc-out", "children"),
          Input("mc-fc-go", "n_clicks"), Input("mc-fc-prod", "value"))
def _forecast(_n, product_id):
    if not product_id:
        return C.empty("Pick a product to forecast.")
    d = api_get("/api/v1/merchandising/demand-forecast", {"product_id": product_id})
    fc = (d or {}).get("forecast") or []
    if not fc:
        return C.empty("No forecast could be produced for this product.")
    hist = d.get("historical") or []

    fig = T.figure(height=280, showlegend=True, margin=dict(l=8, r=8, t=4, b=4))

    fig.add_scatter(x=[p["date"] for p in fc], y=[p["lower_bound"] for p in fc],
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[p["date"] for p in fc], y=[p["upper_bound"] for p in fc],
                    mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(92,143,110,.16)", name="Confidence band",
                    showlegend=True, hoverinfo="skip")

    if hist:
        fig.add_scatter(x=[p["date"] for p in hist], y=[p["actual_qty"] for p in hist],
                        mode="lines+markers", name="Actual",
                        line=dict(color=colors.BRAND, width=2),
                        marker=dict(size=5, color=colors.BRAND),
                        hovertemplate="Actual %{y:.1f}<extra></extra>")

    join = [hist[-1]] if hist else []
    fig.add_scatter(x=[p["date"] for p in join] + [p["date"] for p in fc],
                    y=[p["actual_qty"] for p in join] + [p["predicted_qty"] for p in fc],
                    mode="lines", name="Forecast",
                    line=dict(color=colors.BRAND2, width=2, dash="dot"),
                    hovertemplate="Forecast %{y:.1f}<extra></extra>")

    fig.update_layout(xaxis=dict(type="date", showgrid=False),
                      yaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"], title="Units/day"))

    peak = max(fc, key=lambda p: p.get("predicted_qty", 0)) if fc else {}
    # The series is the point of this card, so it takes the larger share and the
    # summary figures stack beside it rather than above it.
    return [
        C.split(
            [
                C.kpi_grid([
                    C.kpi("30-day forecast", f"{float(d.get('total_forecast', 0) or 0):,.0f}",
                          "units"),
                    C.kpi("Avg daily demand", f"{float(d.get('avg_daily_demand', 0) or 0):,.2f}",
                          "units/day"),
                    C.kpi("Peak day", f"{float(peak.get('predicted_qty', 0)):,.1f}",
                          peak.get("date", "")),
                    C.kpi("Horizon", f"{len(fc)} days", f"{len(hist)} observed points"),
                ]),
                html.Div([C.pill(d.get("model", "—"), "info")], className="row-wrap mt-14"),
            ],
            C.graph(fig, 300),
            weight="wide-right",
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════

def layout():
    """
    Returns the page shell instantly (zero API calls).
    All data-fetching happens in callbacks triggered by dcc.Store('mc-page-load').
    """
    return module_page(
        D.index, D.title, D.summary,
        [
            # This Store fires on every page visit and triggers all lazy-load callbacks.
            dcc.Store(id="mc-page-load", data=1),
            html.Div(
                [
                    _card_pricing(),
                    _card_competitor(),
                    _card_promotion(),
                    _card_assortment(),
                    _card_forecast(),
                ],
                className="grid-2",
            ),
        ],
    )
