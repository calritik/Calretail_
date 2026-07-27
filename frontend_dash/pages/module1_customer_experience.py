"""
Domain 01 — Customer Experience (notebooks 01–04).

Four capability cards — all data-driven, served by FastAPI:
  1. Hyper-personalised Recommendations  (nb 01) — Admin view
  2. Personalised Buying Assistants       (nb 02)
  3. Next-Best-Offer Engines              (nb 03)
  4. Communication Timing Optimiser       (nb 04)
"""
from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from frontend_dash.components import cards as C
from frontend_dash.components.layout import module_page
from frontend_dash.services.api import api_get, api_post
from frontend_dash.services.capabilities import CUSTOMER_EXPERIENCE as D
from frontend_dash.services.capabilities import cap
from frontend_dash.theme import chart_theme as T
from frontend_dash.theme import colors

dash.register_page(__name__, path=D.path, name=D.title)

CAP = {c.key: c for c in D.capabilities}


def _customer_options(limit: int = 60):
    rows = api_get("/api/v1/customer-experience/customers", {"limit": limit}) or []
    return [{"label": f"{r['name']} ({r.get('segment', '—')})",
             "value": r["customer_id"]} for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Hyper-personalised Recommendations  (notebook 01) — Admin View
# ══════════════════════════════════════════════════════════════════════════════

def _card_recommendations(opts):
    return C.card(
        cap("cx", "recommendations").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="cx-rec-cust", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a customer"),
                    dcc.Input(id="cx-rec-n", type="number", value=10, min=1, max=25,
                              className="cp-input", style={"width": "72px"},
                              placeholder="Top-N"),
                    html.Button("Recommend", id="cx-rec-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="cx-rec-out"),
        ],
        caption=(
            "Admin view — shows why each product was recommended: past purchases, "
            "ratings, and what similar shoppers actually bought."
        ),
        info=(
            "<b>Algorithm:</b> user–user collaborative filtering on the purchase matrix. "
            "Products are ranked by how many similar customers bought them + avg rating. "
            "<b>Reasons</b> explain each recommendation in plain English — no raw scores. "
            "<b>Cold-start:</b> falls back to bestsellers in the customer's preferred category."
        ),
        span=2,
    )


@callback(Output("cx-rec-out", "children"),
          Input("cx-rec-go", "n_clicks"), Input("cx-rec-cust", "value"),
          State("cx-rec-n", "value"))
def _recommendations(_n, customer_id, top_n):
    if not customer_id:
        return C.empty("Pick a customer to generate recommendations.")
    data = api_post("/api/v1/customer-experience/recommendations",
                    {"customer_id": customer_id, "top_n": int(top_n or 10)})
    if not data:
        return C.empty("Recommendations unavailable — is the backend running?")

    recs             = data.get("recommendations") or []
    profile          = data.get("profile") or {}
    algo             = data.get("algorithm", "—")
    similar_custs    = data.get("similar_customers") or []
    category_affinity = data.get("category_affinity") or []

    if not recs:
        return C.empty("No recommendations returned for this customer.")

    # ── Customer profile pills ────────────────────────────────────────────────
    segment   = profile.get("segment") or "—"
    loyalty   = profile.get("loyalty_tier") or "—"
    pref_cat  = profile.get("preferred_category") or "—"

    meta_pills = html.Div([
        html.Div("👤 Customer Profile", style={"fontWeight": 600, "marginBottom": "6px",
                                               "fontSize": "12px", "color": colors.INK_SOFT}),
        html.Div([
            C.pill(f"🏷 Segment: {segment}", "info"),
            C.pill(f"⭐ Loyalty: {loyalty}", "neutral"),
            C.pill(f"📦 Preferred: {pref_cat}", "neutral"),
        ], className="row-wrap"),
    ], style={"marginBottom": "14px", "padding": "10px 12px",
              "background": "rgba(92,143,110,.07)", "borderRadius": "8px",
              "border": "1px solid rgba(92,143,110,.18)"})

    # ── "You may like this" product cards ─────────────────────────────────────
    prod_cards = []
    for r in recs:
        reasons     = r.get("reasons") or ["Popular among shoppers"]
        freq        = r.get("freq_by_similar", 0)
        avg_rating  = r.get("avg_rating", 0.0)
        review_cnt  = r.get("review_count", 0)
        price       = r.get("price", 0)

        # Reason tags
        reason_tags = html.Div(
            [html.Span(f"✓ {rsn}", style={
                "fontSize": "10.5px", "color": colors.OK_INK,
                "background": colors.OK_SOFT, "borderRadius": "4px",
                "padding": "2px 7px", "marginRight": "4px", "marginBottom": "3px",
                "display": "inline-block",
            }) for rsn in reasons],
            style={"marginTop": "4px"}
        )

        # Freq badge
        freq_badge = html.Span(
            f"{freq} similar customers bought this",
            style={"fontSize": "10px", "color": colors.BRAND,
                   "fontWeight": 600, "display": "block", "marginTop": "4px"}
        ) if freq > 0 else html.Span()

        # Stars
        full  = int(avg_rating)
        stars = "★" * full + "☆" * (5 - full)
        rating_line = html.Span(
            f"{stars}  {avg_rating:.1f}  ({review_cnt} reviews)",
            style={"fontSize": "11px", "color": colors.WARN_INK}
        )

        card_div = html.Div([
            html.Div([
                html.Div(r.get("product_name", "—"),
                         style={"fontWeight": 700, "fontSize": "13px"}),
                html.Div(f"{r.get('brand', '—')}  ·  {r.get('category', '—')}",
                         className="small muted"),
            ]),
            html.Div([
                html.Div(f"₹{price:,.0f}",
                         style={"fontWeight": 700, "fontSize": "15px", "color": colors.BRAND}),
                rating_line,
                freq_badge,
                reason_tags,
            ], style={"marginTop": "6px"}),
        ], style={
            "padding": "10px 14px",
            "background": "rgba(92,143,110,.04)",
            "border": "1px solid rgba(92,143,110,.15)",
            "borderRadius": "8px",
            "marginBottom": "8px",
        })
        prod_cards.append(card_div)

    products_section = html.Div([
        html.Div(f"🛍️ You may like these · {len(prod_cards)}", style={
            "fontWeight": 700, "fontSize": "13px", "marginBottom": "8px",
            "color": colors.INK,
        }),
        # Top-N is the reader's choice and goes to 25 — scrolled, so a large N
        # can't stretch the card past the diagnostics column beside it.
        html.Div(prod_cards, className="scroll-pane"),
    ], style={"marginBottom": "16px"})

    # ── Graph 1: Why it was recommended — reasons frequency ──────────────────
    reason_counter: dict[str, int] = {}
    for r in recs:
        for rsn in (r.get("reasons") or []):
            reason_counter[rsn] = reason_counter.get(rsn, 0) + 1

    fig_why = T.figure(height=200, margin=dict(l=8, r=8, t=28, b=8))
    if reason_counter:
        labels = list(reason_counter.keys())
        vals   = list(reason_counter.values())
        fig_why.add_bar(
            x=vals, y=labels, orientation="h",
            marker_color=colors.CATEGORICAL[:len(labels)],
            hovertemplate="%{y}<br>%{x} products<extra></extra>",
        )
        fig_why.update_layout(
            title=dict(text="Why were products recommended?",
                       font=dict(size=11.5, color=colors.INK_SOFT)),
            xaxis=dict(title="# of products", showgrid=True,
                       gridcolor=colors.LIGHT["grid"]),
            yaxis=dict(autorange="reversed"),
        )
    why_chart = C.graph(fig_why, 200)

    # ── Graph 2: Similar customers — similarity bar ───────────────────────────
    if similar_custs:
        names  = [c["name"] for c in similar_custs]
        sims   = [c["similarity"] for c in similar_custs]
        bar_colors = T.bar_colors(sims)
        fig_sim = T.figure(height=220, margin=dict(l=8, r=8, t=28, b=8))
        fig_sim.add_bar(
            x=sims, y=names, orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}<br>Similarity: %{x:.3f}<extra></extra>",
        )
        fig_sim.update_layout(
            title=dict(text="Similar customers used for recommendations",
                       font=dict(size=11.5, color=colors.INK_SOFT)),
            xaxis=dict(title="Cosine similarity", range=[0, 1], showgrid=True,
                       gridcolor=colors.LIGHT["grid"]),
            yaxis=dict(autorange="reversed"),
        )
        sim_chart = C.graph(fig_sim, 220)
    else:
        sim_chart = html.Div()

    # ── Graph 3: Category affinity ────────────────────────────────────────────
    if category_affinity:
        cats = [c["category"] for c in category_affinity]
        pcts = [c["purchase_pct"] for c in category_affinity]
        fig_cat = T.figure(height=200, margin=dict(l=8, r=8, t=28, b=8))
        fig_cat.add_bar(
            x=pcts, y=cats, orientation="h",
            marker_color=colors.CATEGORICAL[:len(cats)],
            hovertemplate="%{y}: %{x:.1f}% of purchases<extra></extra>",
            text=[f"{p:.0f}%" for p in pcts],
            textposition="outside",
        )
        fig_cat.update_layout(
            title=dict(text="Customer's purchase history by category",
                       font=dict(size=11.5, color=colors.INK_SOFT)),
            xaxis=dict(title="% of past purchases", showgrid=True,
                       gridcolor=colors.LIGHT["grid"]),
            yaxis=dict(autorange="reversed"),
        )
        cat_chart = C.graph(fig_cat, 200)
    else:
        cat_chart = html.Div()

    # ── Algorithm badge ───────────────────────────────────────────────────────
    algo_badge = html.Div(
        f"⚙ {algo}",
        style={"fontSize": "11px", "color": colors.INK_SOFT, "marginTop": "8px"}
    )

    # The recommendations and the evidence behind them are two parallel readings
    # of the same result, so they sit side by side: what to show the shopper on
    # the left, why the model chose it on the right. Stacked, the three
    # diagnostic charts pushed the product list a full screen out of view.
    return [
        C.split(
            [meta_pills, products_section],
            html.Div([
                html.Div("📊 Admin Diagnostics", style={
                    "fontWeight": 700, "fontSize": "13px",
                    "marginBottom": "12px", "color": colors.INK,
                }),
                why_chart,
                html.Div(style={"height": "12px"}),
                sim_chart,
                html.Div(style={"height": "12px"}),
                cat_chart,
            ], style={
                "padding": "12px 14px",
                "background": "rgba(0,0,0,.025)",
                "borderRadius": "8px",
                "border": "1px solid " + colors.CARD_LINE,
            }),
            ruled=False,
        ),
        algo_badge,
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Personalized Buying Assistants  (notebook 02)
# ══════════════════════════════════════════════════════════════════════════════

SUGGESTED = ["red jackets under 3000", "formal shirts for office",
             "running shoes below 5000", "ethnic wear for a wedding"]


def _card_assistant(opts):
    return C.card(
        cap("cx", "assistant").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="cx-asst-cust", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown",
                                 style={"minWidth": "190px"}),
                    dcc.Input(id="cx-asst-msg", className="cp-input grow", debounce=True,
                              placeholder="Ask for a product in plain English…",
                              value=SUGGESTED[0]),
                    html.Button("Ask", id="cx-asst-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div([html.Span(s, className="chip", id={"type": "cx-chip", "i": i})
                      for i, s in enumerate(SUGGESTED)], className="mb-10"),
            html.Div(id="cx-asst-out"),
        ],
        caption="Natural-language intent extraction — category, colour and price ceiling parsed, matched against live stock.",
        info=(
            "<b>Flow:</b> the LLM (or a rule-based fallback when no key is set) extracts "
            "<b>intent</b>, <b>category</b> and <b>max price</b>, which become a catalogue filter. "
            "Product suggestions come directly from the backend catalogue, not hardcoded lists."
        ),
    )


@callback(Output("cx-asst-msg", "value"),
          Input({"type": "cx-chip", "i": dash.ALL}, "n_clicks"),
          prevent_initial_call=True)
def _chip_to_input(_clicks):
    trig = dash.callback_context.triggered_id
    if isinstance(trig, dict) and trig.get("i") is not None:
        return SUGGESTED[trig["i"]]
    return dash.no_update


@callback(Output("cx-asst-out", "children"),
          Input("cx-asst-go", "n_clicks"), Input("cx-asst-msg", "value"),
          State("cx-asst-cust", "value"))
def _assistant(_n, message, customer_id):
    if not message or not customer_id:
        return C.empty("Ask a question to see how the assistant interprets it.")
    data = api_post("/api/v1/customer-experience/buying-assistant",
                    {"customer_id": customer_id, "message": message})
    if not data:
        return C.empty("Assistant unavailable.")

    chips = []
    if data.get("detected_category"):
        chips.append(C.pill(data["detected_category"], "info"))
    if data.get("max_price"):
        chips.append(C.pill(f"under ₹{data['max_price']:,.0f}", "neutral"))
    if data.get("intent"):
        chips.append(C.pill(f"intent · {data['intent']}", "neutral"))

    out = [
        html.Div([html.Div(message, className="chat-bubble-user copilot-in"),
                  html.Div(data.get("response", "—"), className="chat-bubble-ai copilot-in")],
                 className="chat-stream mb-10"),
        html.Div(chips, className="row-wrap mb-10"),
    ]

    sugg = data.get("product_suggestions") or []
    if sugg:
        out.append(C.table(
            ["Product", "Brand", "Price"],
            [[s.get("product_name", "—"), s.get("brand", "—"),
              f"₹{s.get('price', 0):,.0f}"] for s in sugg],
            numeric={2}, wide={0},
        ))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3 — Next-Best-Offer Engines  (notebook 03)
# ══════════════════════════════════════════════════════════════════════════════

def _segment_options():
    """Fetched live from backend — NOT hardcoded."""
    d = api_get("/api/v1/customer-experience/segmentation") or {}
    return [{"label": s["segment"], "value": s["segment"]} for s in (d.get("segments") or [])]


def _card_nbo(seg_opts):
    return C.card(
        cap("cx", "nbo").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="cx-nbo-seg", options=seg_opts,
                                 value=seg_opts[0]["value"] if seg_opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a segment"),
                    html.Button("Plan offers", id="cx-nbo-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="cx-nbo-out"),
        ],
        caption="The offers most worth pushing to a whole segment, ranked by fit and predicted uplift.",
        info=(
            "<b>Campaign-planning view:</b> active promotions are scored against the segment's "
            "dominant preferred category and channel and whether they on-target the segment, then "
            "ranked. Segments are loaded live from the backend — no hardcoded list."
        ),
    )


@callback(Output("cx-nbo-out", "children"),
          Input("cx-nbo-go", "n_clicks"), Input("cx-nbo-seg", "value"))
def _nbo(_n, segment):
    if not segment:
        return C.empty("Pick a segment to plan its offers.")
    d = api_get("/api/v1/customer-experience/next-best-offer/segment",
                {"segment": segment, "top_n": 6})
    offers = (d or {}).get("offers") or []
    if not offers:
        return C.empty("No active offers matched this segment.")

    chips = [
        C.pill(f"{d['segment_size']:,} customers", "info"),
        C.pill(f"prefers {d.get('preferred_category', '—')}", "neutral"),
        C.pill(f"via {d.get('preferred_channel', '—')}", "neutral"),
    ]
    # An off-target row used to put the segment name *inside* the pill, and a pill
    # can't be narrower than its longest word — "Professional" alone set a ~100px
    # floor on this column and pushed the five columns wider than the card. The
    # verdict stays a pill; the segment it actually targets drops to a sub-line,
    # where it wraps freely.
    rows = [
        [o["promo_type"], C.pill(o["category"], "info"), f"{o['discount_pct']:.0f}%",
         C.pill("on-target", "low") if o["on_target"] else
         html.Div([C.pill("off-target", "neutral"),
                   html.Div(o["target_segment"], className="small muted")],
                  className="cell-stack"),
         f"{o['predicted_uplift_pct']:.0f}%"]
        for o in offers
    ]
    return [
        html.Div(chips, className="row-wrap mb-10"),
        C.table(["Offer", "Category", "Disc.", "Targeting", "Uplift"], rows,
                numeric={2, 4}, wide={0}),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4 — Communication Timing Optimiser  (notebook 04)
# ══════════════════════════════════════════════════════════════════════════════

def _card_comm_timing(opts):
    return C.card(
        cap("cx", "comm_timing").title,
        [
            html.Div(
                [
                    dcc.Dropdown(id="cx-ct-cust", options=opts,
                                 value=opts[0]["value"] if opts else None,
                                 clearable=False, className="dash-dropdown grow",
                                 placeholder="Select a customer"),
                    html.Button("Analyse", id="cx-ct-go", className="cp-go", n_clicks=0),
                ],
                className="cp-row",
            ),
            html.Div(id="cx-ct-out"),
        ],
        caption=(
            "Optimal send-time, day and channel for each customer, derived from their real "
            "browsing activity — no assumed schedules."
        ),
        info=(
            "<b>Source:</b> notebook 04 aggregates the customer's session log by hour and "
            "day-of-week to surface the window when they are most active online. "
            "<b>Channel</b> is inferred from device and session patterns — all live data, "
            "nothing hardcoded."
        ),
        # Spans the grid so the page ends on a full-width row instead of a lone
        # half-width card beside an empty column — and so the hourly histogram
        # gets enough width for 24 legible hour labels.
        span=2,
    )


@callback(Output("cx-ct-out", "children"),
          Input("cx-ct-go", "n_clicks"), State("cx-ct-cust", "value"))
def _comm_timing(_n, customer_id):
    if not customer_id:
        return C.empty("Select a customer to optimise their communication timing.")
    data = api_get("/api/v1/customer-experience/communication-timing",
                   {"customer_id": customer_id})
    if not data:
        return C.empty("Communication timing unavailable — is the backend running?")

    hourly = data.get("hourly_activity") or {}
    if hourly:
        hours_sorted = sorted(hourly.items(), key=lambda kv: int(kv[0]))
        hours = [f"{int(h):02d}:00" for h, _ in hours_sorted]
        pcts  = [round(v * 100, 1) for _, v in hours_sorted]
        fig = T.figure(height=180, margin=dict(l=8, r=8, t=4, b=4))
        fig.add_bar(x=hours, y=pcts, name="Activity",
                    marker_color=colors.BRAND, width=0.65,
                    hovertemplate="%{x}<br>%{y:.1f}% of sessions<extra></extra>")
        fig.update_layout(
            xaxis=dict(title="Hour of day", showgrid=False),
            yaxis=dict(title="% sessions", showgrid=True, gridcolor=colors.LIGHT["grid"]),
        )
        chart = C.graph(fig, 220)
    else:
        chart = C.empty("No session history to derive an activity profile from.")

    open_rate = data.get("predicted_open_rate")
    open_pct  = f"{open_rate * 100:.1f}%" if open_rate is not None else "—"

    # The four recommendations are the answer; the histogram is the evidence.
    # Side by side, one explains the other without needing to be scrolled to.
    return [
        C.split(
            C.kpi_grid([
                C.kpi("Best send time", data.get("best_send_hour_label", "—")),
                C.kpi("Best day",       data.get("best_day_of_week", "—")),
                C.kpi("Recommended channel", data.get("recommended_channel", "—")),
                C.kpi("Predicted open rate", open_pct),
            ]),
            [C.subhead("When this customer is actually online"), chart],
            weight="wide-right",
        ),
    ]


# ══════════════════════════════════════════════════════════════════════════════

def layout():
    cust_opts = _customer_options()
    seg_opts  = _segment_options()
    banner = [] if cust_opts else [C.offline_banner()]
    return module_page(
        D.index, D.title, D.summary,
        banner + [
            html.Div(
                [
                    _card_recommendations(cust_opts),
                    _card_nbo(seg_opts),
                    _card_assistant(cust_opts),
                    _card_comm_timing(cust_opts),
                ],
                className="grid-2",
            )
        ],
    )
