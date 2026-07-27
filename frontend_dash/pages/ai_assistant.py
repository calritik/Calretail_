"""
AI Assistant — natural-language front door to all four domains.

`POST /api/v1/support/assistant-router` classifies a free-text question into an
action and extracts any customer/product id it can see, but its
vocabulary is narrow (deterministic keyword fallback when no LLM key is
configured, which is most of the time right now — the Gemini key is
rate-limited). To let this page "know about" the full deck rather than a
handful of endpoints, a local, deterministic keyword table is checked first,
built from real trigger phrases tied to each of the platform's 16 capabilities
across all four domains. Only when nothing local matches does the page fall
back to the router's narrower action set, and only when that also comes back
"general_chat" does it show the router's own friendly response.

Every answer here is a live call to the same FastAPI endpoints the four domain
pages use — nothing is fabricated. And nothing on this page ever renders a raw
entity code (``C00001``, ``P00001``, ...): ids are only ever used as call
arguments, never as visible text. Where an id is required and the question
didn't name one, a real entity is picked from the corresponding list endpoint
(first row) and named — never a made-up id.
"""
from __future__ import annotations

import re

import dash
from dash import Input, State, Output, callback, dcc, html

from frontend_dash.components import cards as C
from frontend_dash.components.layout import module_page
from frontend_dash.services.api import api_get, api_post
from frontend_dash.theme import chart_theme as T
from frontend_dash.theme import colors

dash.register_page(__name__, path="/ai-assistant", name="AI Assistant")

EXAMPLES = [
    "Recommend products for a loyal customer",
    "Which SKUs are running low on stock?",
    "Forecast demand for a top seller",
    "Optimize a delivery route",
    "Which products should we drop from assortment?",
]


def _normalise_id(raw: str | None, prefix: str, width: int = 5) -> str | None:
    """
    The router's regex fallback emits ids like ``P-0001`` / ``CUST-00001``,
    while the data uses ``P00001`` / ``C00001``. Reduce either spelling to the
    digits and re-pad, so an id the router extracted actually resolves.
    """
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    return f"{prefix}{int(digits):0{width}d}" if digits else None


# ── Entity resolution ────────────────────────────────────────────────────────
# Names, never codes. Ids extracted from free text are looked up in the same
# list endpoints the domain pages populate their dropdowns from; if an id
# isn't found there (the customer/product catalogues run into the thousands
# and the list endpoints cap out well before that) the chip is simply omitted
# — never shown as a raw code. Where a capability needs an entity the question
# didn't name, the first row of the real list endpoint is picked and named in
# the answer.

def _customer_name(cid: str | None) -> str | None:
    if not cid:
        return None
    rows = api_get("/api/v1/customer-experience/customers", {"limit": 500}) or []
    return next((r.get("name") for r in rows if r.get("customer_id") == cid), None)


def _product_name(pid: str | None) -> str | None:
    if not pid:
        return None
    rows = api_get("/api/v1/merchandising/products", {"limit": 200}) or []
    return next((r.get("product_name") for r in rows if r.get("product_id") == pid), None)


def _location_name(loc_id: str | None) -> str | None:
    if not loc_id:
        return None
    stores = api_get("/api/v1/operations/stores", {"limit": 200}) or []
    hit = next((r.get("store_name") for r in stores if r.get("store_id") == loc_id), None)
    if hit:
        return hit
    whs = api_get("/api/v1/operations/warehouses") or []
    return next((r.get("warehouse_name") for r in whs if r.get("warehouse_id") == loc_id), None)


def _default_customer():
    rows = api_get("/api/v1/customer-experience/customers", {"limit": 1}) or []
    return (rows[0]["customer_id"], rows[0]["name"]) if rows else (None, None)


def _default_product():
    rows = api_get("/api/v1/merchandising/products", {"limit": 1}) or []
    return (rows[0]["product_id"], rows[0]["product_name"]) if rows else (None, None)


def _default_warehouse():
    rows = api_get("/api/v1/operations/warehouses") or []
    return (rows[0]["warehouse_id"], rows[0]["warehouse_name"]) if rows else (None, None)




def _default_promo():
    rows = api_get("/api/v1/merchandising/promotions", {"limit": 1}) or []
    return (rows[0]["promo_id"], f"{rows[0]['promo_type']} promotion") if rows else (None, None)


def _resolve_or_pick(id_val, default_fn, kind: str):
    """Uses the given id if present; otherwise picks a real entity from its
    list endpoint and returns a note naming what was picked (never its id)."""
    if id_val:
        return id_val, None
    picked_id, picked_name = default_fn()
    if not picked_id:
        return None, None
    note = html.Div(
        ["Showing ", kind, " for ", html.B(picked_name), " — none named in your question."],
        className="small muted mb-8",
    )
    return picked_id, note


# ── Per-capability executors ─────────────────────────────────────────────────
# Each returns the rendered answer for its capability. They deliberately reuse
# the same endpoints the domain pages use, so the assistant can never drift
# from what those pages would show for the same entity. Signature is uniform
# — (cid, pid, query) — even where an executor only needs one or none of them.

# — Domain 01 · Customer Experience ------------------------------------------

def _run_recommendations(cid, _pid, _q):
    cid, note = _resolve_or_pick(cid, _default_customer, "recommendations")
    if not cid:
        return C.empty("No customer available to recommend against.")
    d = api_post("/api/v1/customer-experience/recommendations", {"customer_id": cid, "top_n": 6})
    recs = (d or {}).get("recommendations") or []
    if not recs:
        return [note] if note else C.empty("No recommendations available for that customer.")
    # No Score column: a bare 0.184 tells the reader nothing they can act on,
    # and this page's whole premise is that it never surfaces raw model output.
    # Dropping it also gives the product name the width it was competing for.
    table = C.table(
        ["Product", "Category", "Price"],
        [[r.get("product_name", "—"), C.pill(r.get("category", "—"), "info"),
          f"₹{r.get('price', 0):,.0f}"] for r in recs],
        numeric={2}, wide={0},
    )
    return [note, table] if note else table


def _run_next_best_offer(cid, _pid, _q):
    cid, note = _resolve_or_pick(cid, _default_customer, "the next-best offer")
    if not cid:
        return C.empty("No customer available to plan an offer for.")
    d = api_get("/api/v1/customer-experience/next-best-offer", {"customer_id": cid})
    if not d:
        return C.empty("No offer could be found for that customer.")
    uplift = float(d.get("predicted_uplift", 0) or 0)
    out = [
        C.kpi_grid([
            C.kpi("Offer", d.get("promo_type", "—")),
            C.kpi("Discount", f"{d.get('discount_pct', 0):.0f}%"),
            C.kpi("Predicted uplift", f"{uplift:.0f}%", "", "up" if uplift >= 0 else "down"),
            C.kpi("Confidence", f"{float(d.get('confidence', 0) or 0) * 100:.0f}%"),
        ]),
    ]
    return [note] + out if note else out


def _run_comm_timing(cid, _pid, _q):
    cid, note = _resolve_or_pick(cid, _default_customer, "communication timing")
    if not cid:
        return C.empty("No customer available for communication timing.")
    d = api_get("/api/v1/customer-experience/communication-timing", {"customer_id": cid})
    if not d:
        return [note] if note else C.empty("Communication timing unavailable.")
    open_rate = d.get("predicted_open_rate")
    out = [
        C.kpi_grid([
            C.kpi("Best send time",  d.get("best_send_hour_label", "\u2014")),
            C.kpi("Best day",        d.get("best_day_of_week", "\u2014")),
            C.kpi("Channel",         d.get("recommended_channel", "\u2014")),
            C.kpi("Pred. open rate", f"{open_rate * 100:.1f}%" if open_rate else "\u2014"),
        ]),
    ]
    return [note] + out if note else out


def _run_buying_assistant(cid, _pid, query):
    cid, note = _resolve_or_pick(cid, _default_customer, "the buying assistant")
    if not cid:
        return C.empty("No customer available to run the assistant for.")
    d = api_post("/api/v1/customer-experience/buying-assistant", {"customer_id": cid, "message": query})
    if not d:
        return C.empty("Assistant unavailable.")
    chips = []
    if d.get("detected_category"):
        chips.append(C.pill(d["detected_category"], "info"))
    if d.get("max_price"):
        chips.append(C.pill(f"under ₹{d['max_price']:,.0f}", "neutral"))
    sugg = d.get("product_suggestions") or []
    out = [
        html.Div(d.get("response", "—"), className="chat-bubble-ai copilot-in"),
        html.Div(chips, className="row-wrap mt-8 mb-10") if chips else None,
    ]
    if sugg:
        out.append(C.table(
            ["Product", "Brand", "Price"],
            [[s.get("product_name", "—"), s.get("brand", "—"), f"₹{s.get('price', 0):,.0f}"] for s in sugg],
            numeric={2},
        ))
    return [note] + out if note else out


# — Domain 02 · Merchandising -------------------------------------------------

def _run_forecast(_cid, pid, _q):
    pid, note = _resolve_or_pick(pid, _default_product, "the demand forecast")
    if not pid:
        return C.empty("No product available to forecast.")
    d = api_get("/api/v1/merchandising/demand-forecast", {"product_id": pid})
    if not d or not d.get("forecast"):
        return [note] if note else C.empty("No forecast available for that product.")
    f = d["forecast"]
    fig = T.figure(height=200)
    fig.add_scatter(x=[p["date"] for p in f], y=[p["predicted_qty"] for p in f],
                    mode="lines", line=dict(color=colors.BRAND, width=2.2),
                    name="Forecast", hovertemplate="%{x}<br>%{y:.1f} units<extra></extra>")
    fig.update_layout(yaxis=dict(showgrid=True, gridcolor=colors.LIGHT["grid"]))
    out = [
        C.kpi_grid([
            C.kpi("30-day forecast", f"{d.get('total_forecast', 0):,.0f}", "units"),
            C.kpi("Avg daily", f"{d.get('avg_daily_demand', 0):,.2f}"),
            C.kpi("Model", d.get("model", "—")),
        ]),
        html.Div(C.graph(fig, 200), className="mt-14"),
    ]
    return [note] + out if note else out


def _run_dynamic_pricing(_cid, pid, _q):
    pid, note = _resolve_or_pick(pid, _default_product, "dynamic pricing")
    if not pid:
        return C.empty("No product available to price.")
    d = api_post("/api/v1/merchandising/dynamic-pricing", {"product_id": pid})
    if not d:
        return [note] if note else C.empty("No price recommendation returned for that product.")
    delta = float(d.get("price_delta_pct", 0) or 0)
    lift = float(d.get("expected_revenue_lift_pct", 0) or 0)
    out = [
        C.kpi_grid([
            C.kpi("Current price", f"₹{float(d.get('current_price', 0) or 0):,.0f}"),
            C.kpi("Recommended price", f"₹{float(d.get('recommended_price', 0) or 0):,.0f}",
                  f"{delta:+.2f}% vs current", "up" if delta >= 0 else "down"),
            C.kpi("Expected revenue lift", f"{lift:+.1f}%", "", "up" if lift >= 0 else "down"),
        ]),
        html.Div(d.get("rationale", ""), className="small muted mt-8"),
    ]
    return [note] + out if note else out


def _run_competitor_monitoring(_cid, _pid, _q):
    rows = (api_get("/api/v1/merchandising/competitor-monitoring") or {}).get("results") or []
    if not rows:
        return C.empty("Competitor monitoring feed unavailable.")
    total = len(rows)
    alerts = [r for r in rows if r.get("alert_flag")]
    worst = sorted(alerts, key=lambda r: -abs(float(r.get("price_gap_pct", 0) or 0)))[:6]
    tbl = [[r.get("product_name", "—"), f"₹{float(r.get('our_price', 0) or 0):,.0f}",
            f"{float(r.get('price_gap_pct', 0) or 0):+.1f}%", C.pill(r.get("status", "—"), r.get("status", ""))]
           for r in worst]
    return [
        C.kpi_grid([
            C.kpi("SKUs monitored", f"{total:,}"),
            C.kpi("Price alerts", f"{len(alerts):,}", f"{len(alerts) / total * 100:.1f}% of the sweep", "down"),
        ]),
        html.Div(C.table(["Product", "Our price", "Gap", "Status"], tbl, numeric={1, 2}), className="mt-14"),
    ]


def _run_promotion_optimization(_cid, _pid, _q):
    promo_id, note = _resolve_or_pick(None, _default_promo, "promotion evaluation")
    if not promo_id:
        return C.empty("No promotion available to evaluate.")
    d = api_get("/api/v1/merchandising/promotion-optimization", {"promo_id": promo_id})
    if not d:
        return [note] if note else C.empty("No experiment result available for that promotion.")
    uplift = float(d.get("uplift_pct", 0) or 0)
    out = [
        C.kpi_grid([
            C.kpi("Control revenue", C.money(d.get("control_revenue", 0))),
            C.kpi("Treated revenue", C.money(d.get("treated_revenue", 0)), f"{uplift:+.1f}% uplift",
                  "up" if uplift >= 0 else "down"),
            C.kpi("Incremental", C.money(d.get("incremental_revenue", 0)), "net of control", "up"),
        ]),
    ]
    return [note] + out if note else out


# — Domain 03 · Operational Efficiency ---------------------------------------

def _run_inventory(_cid, _pid, _q):
    rows = (api_get("/api/v1/operations/inventory-health") or {}).get("results") or []
    if not rows:
        return C.empty("Inventory health unavailable.")
    worst = sorted(rows, key=lambda r: r.get("health_score", 1))[:6]
    at_risk = sum(1 for r in rows if (r.get("risk_label") or "").lower() != "healthy")
    tbl = []
    for r in worst:
        where = r.get("location_name") or _location_name(r.get("store_id") or r.get("warehouse_id")) or "—"
        tbl.append([r.get("product_name", "—"), where, f"{r.get('stock_qty', 0):,}",
                    f"{r.get('days_cover', 0):,.0f}", C.pill(r.get("risk_label", "—"), r.get("risk_label", ""))])
    return [
        C.kpi_grid([
            C.kpi("SKU-locations", f"{len(rows):,}"),
            C.kpi("At risk", f"{at_risk:,}", "not healthy", "down" if at_risk else ""),
        ]),
        html.Div(C.table(["Product", "Location", "Stock", "Days cover", "Risk"], tbl, numeric={2, 3}),
                 className="mt-14"),
    ]


def _run_replenishment(_cid, _pid, _q):
    d = api_get("/api/v1/operations/replenishment")
    rows = (d or {}).get("replenishment_orders") or []
    if not rows:
        return C.empty("No replenishment orders pending.")
    tbl = [[r.get("product_name", "—"), f"{r.get('qty', 0):,}", r.get("supplier", "—"),
            r.get("priority", "—")] for r in rows[:6]]
    return [
        C.kpi_grid([
            C.kpi("Orders pending", f"{d.get('total_orders', 0):,}"),
            C.kpi("Total value", C.money(d.get('total_value', 0))),
        ]),
        html.Div(C.table(["Product", "Qty", "Supplier", "Priority"], tbl, numeric={1}), className="mt-14"),
    ]


def _run_warehouse_optimization(_cid, _pid, _q):
    wh_id, note = _resolve_or_pick(None, _default_warehouse, "the warehouse slotting plan")
    if not wh_id:
        return C.empty("No warehouse available to slot.")
    d = api_get("/api/v1/operations/warehouse-optimization", {"warehouse_id": wh_id})
    if not d:
        return [note] if note else C.empty("No slotting plan returned for that warehouse.")
    summary = d.get("class_summary") or {}
    plan = d.get("slotting_plan") or []
    total = sum(int(v or 0) for v in summary.values()) or 1
    out = [
        C.kpi_grid([
            C.kpi("Pick time reduction", f"{float(d.get('estimated_pick_time_reduction_pct') or 0):.1f}%",
                  "", "up"),
            C.kpi("SKUs slotted", f"{len(plan):,}"),
            C.kpi("Class A share", f"{int(summary.get('A', 0)) / total * 100:.0f}%" if summary else "—"),
        ]),
    ]
    return [note] + out if note else out


def _run_route_optimization(_cid, _pid, _q):
    wh_id, note = _resolve_or_pick(None, _default_warehouse, "the delivery route")
    if not wh_id:
        return C.empty("No warehouse available to route from.")
    d = api_post("/api/v1/operations/route-optimization", {"warehouse_id": wh_id})
    if not d or not d.get("route"):
        return [note] if note else C.empty("No open deliveries to route from that warehouse.")
    saving = float(d.get("saving_pct", 0) or 0)
    out = [
        C.kpi_grid([
            C.kpi("Optimised distance", f"{float(d.get('optimised_distance_km') or 0):,.0f} km"),
            C.kpi("Distance saved", f"{float(d.get('distance_saved_km') or 0):,.0f} km",
                  f"{saving:.1f}% shorter", "up"),
            C.kpi("Drive time", f"{float(d.get('estimated_time_hrs') or 0):,.1f} hrs"),
            C.kpi("Orders on board", f"{d.get('total_orders', 0):,}"),
        ]),
        html.Div(C.bar_row("Distance saved vs. baseline", f"{saving:.1f}%", saving,
                           "ok" if saving >= 20 else "warn" if saving >= 8 else "danger"),
                 className="mt-14"),
    ]
    return [note] + out if note else out


# — Domain 04 · Customer Support ----------------------------------------------

def _run_triage(cid, _pid, query=""):
    cid, note = _resolve_or_pick(cid, _default_customer, "ticket triage")
    d = api_post("/api/v1/support/ticket-triage",
                 {"ticket_description": query, "customer_id": cid})
    if not d:
        return [note] if note else C.empty("Triage unavailable.")
    pr = d.get("assigned_priority", "—")
    out = [
        html.Div([C.pill(d.get("predicted_category", "—"), "info"), C.pill(pr, pr),
                  C.pill(d.get("routing_department", "—"), "neutral")], className="row-wrap"),
        html.Div(C.bar_row("Confidence", f"{d.get('overall_confidence', 0) * 100:.0f}%",
                           d.get("overall_confidence", 0) * 100,
                           "ok" if d.get("overall_confidence", 0) >= .6 else "warn"),
                 className="mt-14"),
    ]
    return [note] + out if note else out




def _run_chatbot(cid, _pid, query):
    cid, note = _resolve_or_pick(cid, _default_customer, "the chatbot session")
    if not cid:
        return C.empty("No customer available to open a chatbot session for.")
    d = api_post("/api/v1/support/chatbot", {"customer_id": cid, "message": query})
    if not d:
        return [note] if note else C.empty("Chatbot unavailable.")
    escalate = bool(d.get("escalate"))
    out = [
        html.Div(d.get("response", "—"), className="chat-bubble-ai copilot-in"),
        html.Div([C.pill(f"intent · {d.get('intent', '—')}", "info"),
                  C.pill("escalate · yes" if escalate else "escalate · no", "high" if escalate else "low")],
                 className="row-wrap mt-8"),
    ]
    return [note] + out if note else out


def _run_agent_assist(cid, _pid, query):
    cid, note = _resolve_or_pick(cid, _default_customer, "agent assist")
    if not cid:
        return C.empty("No customer available for agent assist.")
    d = api_post("/api/v1/support/agent-assist", {"query_text": query, "customer_id": cid})
    if not d:
        return [note] if note else C.empty("Agent assist unavailable.")
    conf = float(d.get("confidence", 0) or 0)
    response = d.get("suggested_response") or d.get("resolution", "\u2014")
    sop = d.get("matched_sop") or d.get("sop", "")
    parts = []
    if sop:
        parts.append(html.Div(sop, className="chat-bubble-ai copilot-in mb-6"))
    parts.append(html.Div(response, className="chat-bubble-ai copilot-in"))
    if conf:
        tone = "ok" if conf >= 0.7 else "warn" if conf >= 0.4 else "danger"
        parts.append(C.bar_row("Retrieval confidence", f"{conf:.0%}", conf * 100, tone))
    return [note] + parts if note else parts


def _run_voc(_cid, pid, _q):
    params = {"product_id": pid} if pid else {}
    d = api_get("/api/v1/support/voice-of-customer", params or None)
    if not d or not d.get("total_reviews"):
        return C.empty("No reviews available for that product.")
    total = d.get("total_reviews", 0)
    avg_r = float(d.get("avg_rating", 0))
    sent  = d.get("sentiment_distribution") or {}
    aspects = d.get("aspect_analysis") or {}
    parts = [
        C.kpi_grid([
            C.kpi("Reviews", f"{total:,}"),
            C.kpi("Avg rating", f"{avg_r:.2f} / 5.0", "",
                  "down" if d.get("alert") else "up"),
            C.kpi("Positive", f"{sent.get('Positive', 0):,}"),
            C.kpi("Negative", f"{sent.get('Negative', 0):,}", "", "down" if sent.get("Negative", 0) else ""),
        ]),
    ]
    if aspects:
        tbl = [[asp, f"{info['mention_count']:,}", f"{info['avg_rating']:.2f}",
                f"{info['pct_positive']:.0f}%"]
               for asp, info in sorted(aspects.items(),
                                       key=lambda kv: kv[1]["mention_count"], reverse=True)[:6]]
        parts.append(html.Div(C.table(["Aspect", "Mentions", "Avg rating", "% Positive"],
                                      tbl, numeric={1, 2, 3}), className="mt-14"))
    return parts



# ── Dispatch table ────────────────────────────────────────────────────────────
# key -> title, owning domain (label, path), trigger keywords, executor. Trigger
# keywords are simple lowercase substrings scored by count — no LLM needed, so
# this table works identically whether or not a Gemini key is configured.

_CX = ("Domain 01", "/customer-experience")
_MERCH = ("Domain 02", "/merchandising")
_OPS = ("Domain 03", "/operations")
_SUPPORT = ("Domain 04", "/support")

CAPS = {
    # Domain 01 — Customer Experience
    "recommendations": {
        "title": "Hyper-personalized Recommendations", "domain": _CX,
        "keywords": ["recommend", "recommendation", "suggest product", "what should i buy"],
        "run": _run_recommendations,
    },
    "buying_assistant": {
        "title": "Personalized Buying Assistants", "domain": _CX,
        "keywords": ["buying assistant", "find me a product", "help me find", "personal shopper",
                     "conversational assistant", "shopping assistant"],
        "run": _run_buying_assistant,
    },
    "next_best_offer": {
        "title": "Next-Best-Offer Engines", "domain": _CX,
        "keywords": ["next best offer", "best offer for", "nbo", "offer to send"],
        "run": _run_next_best_offer,
    },
    "comm_timing": {
        "title": "Communication Timing Optimiser", "domain": _CX,
        "keywords": ["communication timing", "best time to send", "when to email", "optimal send time",
                     "open rate", "best channel"],
        "run": _run_comm_timing,
    },

    # Domain 02 — Merchandising
    "demand_forecast": {
        "title": "Demand Forecasting", "domain": _MERCH,
        "keywords": ["forecast demand", "demand forecast", "sales forecast", "forecast for",
                     "how many units", "top seller"],
        "run": _run_forecast,
    },
    "dynamic_pricing": {
        "title": "Dynamic Pricing Engines", "domain": _MERCH,
        "keywords": ["dynamic pricing", "price recommendation", "what price should",
                     "optimal price", "reprice"],
        "run": _run_dynamic_pricing,
    },
    "promotion_optimization": {
        "title": "Promotion Optimization", "domain": _MERCH,
        "keywords": ["promotion optimization", "promo uplift", "cannibalization",
                     "evaluate promotion", "promotion roi"],
        "run": _run_promotion_optimization,
    },
    "competitor_monitoring": {
        "title": "Competitor Price Monitoring", "domain": _MERCH,
        "keywords": ["competitor price", "competitor monitoring", "underpriced",
                     "price gap", "overpriced"],
        "run": _run_competitor_monitoring,
    },

    # Domain 03 — Operational Efficiency (notebooks 09–12)
    "inventory_health": {
        "title": "Smart Inventory Management", "domain": _OPS,
        "keywords": ["low on stock", "running low", "stock level", "inventory health",
                     "stock-out", "stockout", "which items are low"],
        "run": _run_inventory,
    },
    "replenishment": {
        "title": "Automated Replenishment", "domain": _OPS,
        "keywords": ["replenish", "reorder", "automated replenishment", "restock"],
        "run": _run_replenishment,
    },
    "warehouse_optimization": {
        "title": "Warehouse Optimization", "domain": _OPS,
        "keywords": ["warehouse slotting", "pick time", "warehouse optimization",
                     "picking productivity", "slotting plan"],
        "run": _run_warehouse_optimization,
    },
    "route_optimization": {
        "title": "Logistics, Route & Fleet Optimization", "domain": _OPS,
        "keywords": ["delivery route", "route optimization", "optimize a route", "optimise route",
                     "fleet optimization", "warehouse route"],
        "run": _run_route_optimization,
    },

    # Domain 04 — Customer Support (notebooks 13–16)
    "chatbot": {
        "title": "24x7 AI Chatbots", "domain": _SUPPORT,
        "keywords": ["chatbot", "ask the chatbot", "24x7 support", "order tracking chat", "support bot"],
        "run": _run_chatbot,
    },
    "ticket_triage": {
        "title": "Intelligent Ticket Triage", "domain": _SUPPORT,
        "keywords": ["ticket", "complaint", "triage", "damaged", "replacement", "order arrived"],
        "run": _run_triage,
    },
    "agent_assist": {
        "title": "Agent Assist", "domain": _SUPPORT,
        "keywords": ["agent assist", "live agent", "agent suggestion", "resolution suggestion",
                     "sop lookup", "agent support"],
        "run": _run_agent_assist,
    },
    "voc": {
        "title": "Voice of Customer", "domain": _SUPPORT,
        "keywords": ["voice of customer", "sentiment", "product reviews", "review analysis",
                     "aspect analysis", "customer feedback"],
        "run": _run_voc,
    },
}


def _match_capability(query: str) -> str | None:
    """Deterministic, LLM-free keyword scoring over the full capability table."""
    q = query.lower()
    scores = {key: sum(1 for kw in spec["keywords"] if kw in q) for key, spec in CAPS.items()}
    scores = {k: v for k, v in scores.items() if v > 0}
    if not scores:
        return None
    return max(scores, key=scores.get)


# ══════════════════════════════════════════════════════════════════════════════

@callback(Output("as-query", "value"),
          Input({"type": "as-chip", "i": dash.ALL}, "n_clicks"),
          prevent_initial_call=True)
def _chip_to_input(_clicks):
    trig = dash.callback_context.triggered_id
    if isinstance(trig, dict) and trig.get("i") is not None:
        return EXAMPLES[trig["i"]]
    return dash.no_update


@callback(Output("as-out", "children"),
          Input("as-go", "n_clicks"), State("as-query", "value"))
def _route(_n, query):
    if not query or not query.strip():
        return C.empty("Ask anything — the assistant knows every capability across all four domains.")

    r = api_post("/api/v1/support/assistant-router", {"query": query}) or {}
    cid = _normalise_id(r.get("customer_id"), "C")
    pid = _normalise_id(r.get("product_id"), "P")

    key = _match_capability(query)
    if key is None and r.get("action") in CAPS:
        key = r.get("action")

    header = [
        html.Div([html.Div(query, className="chat-bubble-user copilot-in")], className="chat-stream mb-10"),
    ]

    if key is None:
        chips = [C.pill("Overview", "info"), C.pill("General assistance", "neutral")]
        header.append(html.Div(chips, className="row-wrap mb-10"))
        answer = html.Div(
            r.get("response") or "Ask about recommendations, pricing, inventory, forecasting, routing, "
                                  "assortment, support tickets or any of the platform's other capabilities.",
            className="chat-bubble-ai copilot-in",
        )
        return header + [html.Div(answer)]

    spec = CAPS[key]
    domain_label, domain_path = spec["domain"]
    chips = [C.pill(domain_label, "info"), C.pill(spec["title"], "neutral")]

    cname = _customer_name(cid)
    pname = _product_name(pid)
    if cname:
        chips.append(C.pill(f"customer · {cname}", "neutral"))
    if pname:
        chips.append(C.pill(f"product · {pname}", "neutral"))
    header.append(html.Div(chips, className="row-wrap mb-10"))

    answer = spec["run"](cid, pid, query)

    return header + [
        html.Div(answer),
        html.Div(dcc.Link(f"Open {domain_label} · {spec['title']} →", href=domain_path, className="chip"),
                 className="mt-14"),
    ]


def layout():
    return module_page(
        "Copilot",
        "AI Assistant",
        "One question, routed to the right capability. A deterministic keyword matcher covers every "
        "capability across the four domains — no LLM required — and falls back to the router's "
        "own classifier and friendly chat for anything conversational.",
        [
            html.Div(
                [
                    C.card(
                        "Ask the platform",
                        [
                            html.Div(
                                [
                                    dcc.Input(id="as-query", className="cp-input grow",
                                              placeholder="e.g. Forecast demand for a top seller",
                                              value=EXAMPLES[0], debounce=True),
                                    html.Button("Ask", id="as-go", className="cp-go", n_clicks=0),
                                ],
                                className="cp-row",
                            ),
                            html.Div([html.Span(e, className="chip",
                                                id={"type": "as-chip", "i": i})
                                      for i, e in enumerate(EXAMPLES)], className="mb-10"),
                            html.Div(id="as-out"),
                        ],
                        caption="A local keyword table checked first covers every capability on the console; the "
                                "router's own classifier is the fallback for anything it doesn't recognise.",
                        cls="card-solo",
                        info="<b>Routing:</b> a deterministic keyword matcher built from every capability's "
                             "real trigger phrases runs first, so the assistant works the same whether or "
                             "not an LLM key is configured. Only genuinely conversational questions fall "
                             "through to the router's own (narrower) classifier.",
                        span=2,
                    ),
                ],
                className="grid-2",
            )
        ],
    )
