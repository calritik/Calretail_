"""
Module 1 — Customer Experience Router
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.services.customer_experience import (
    get_recommendations, buying_assistant_query,
    get_next_best_offer, get_communication_timing,
    get_recommendations_debug,
)

router = APIRouter(prefix="/api/v1/customer-experience", tags=["Customer Experience"])


# ── Schemas ──────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    customer_id: str
    top_n: int = 10

class AssistantRequest(BaseModel):
    customer_id: str
    message: str
    session_id: str = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/recommendations")
def recommendations(req: RecommendRequest):
    """Admin view: personalised recommendations with reasons, ratings, and similar-customer data."""
    return get_recommendations_debug(req.customer_id, req.top_n)


@router.get("/recommendations/debug")
def recommendations_debug(customer_id: str = Query(...), top_n: int = Query(10, le=50)):
    """Admin-only: full internal breakdown of the recommendation engine."""
    return get_recommendations_debug(customer_id, top_n)


@router.post("/buying-assistant")
def buying_assistant(req: AssistantRequest):
    """Conversational buying assistant — parse intent and suggest products."""
    return buying_assistant_query(req.customer_id, req.message)


@router.get("/next-best-offer/segment")
def next_best_offer_segment(segment: str = Query(...), top_n: int = Query(6)):
    """Best offers for an entire customer segment."""
    from backend.utils.data_loader import get_customers, get_promotions, get_products
    import pandas as pd
    custs  = get_customers()
    promos = get_promotions()
    prods  = get_products()

    seg_custs = custs[custs["segment"] == segment]
    seg_size  = len(seg_custs)
    if seg_size == 0:
        raise HTTPException(status_code=404, detail=f"Segment '{segment}' not found.")

    pref_cat = seg_custs["preferred_category"].mode().iloc[0] if "preferred_category" in seg_custs.columns else "—"
    pref_ch  = seg_custs["preferred_channel"].mode().iloc[0]  if "preferred_channel"  in seg_custs.columns else "—"

    # Join with products to get category
    prod_cat = prods[["product_id", "category"]].drop_duplicates("product_id")
    active = promos.copy()
    if "is_active" in active.columns:
        active = active[active["is_active"] == True]
    active = active.merge(prod_cat, on="product_id", how="left")

    # discount_pct is stored as decimal (0.05–0.60); convert to % (5–60)
    active["disc_pct"] = active["discount_pct"].astype(float) * 100

    # Data-driven targeting flags
    active["on_target"] = active["target_segment"].astype(str) == str(segment)
    active["cat_match"] = active["category"].astype(str) == str(pref_cat)

    # Retail-realistic uplift model (always < discount, varies per promo):
    #   base     = 35% of discount — industry rule-of-thumb conversion rate
    #              (a 40% discount typically yields ~14% sales volume lift)
    #   on-target= flat +6 pts  (right audience → higher uptake)
    #   cat_match= flat +3 pts  (preferred category → even higher engagement)
    #   duration = small decay for very long promos (urgency fades)
    dur = active["duration_days"].astype(float) if "duration_days" in active.columns \
          else 14.0
    dur_factor = (14.0 / dur.clip(lower=1)).clip(upper=1.5)   # shorter promos feel more urgent

    active["predicted_uplift_pct"] = (
        active["disc_pct"] * 0.35 * dur_factor
        + active["on_target"].astype(float) * 6.0
        + active["cat_match"].astype(float) * 3.0
    ).round(1).clip(upper=active["disc_pct"] * 0.75)  # uplift always < 75% of discount


    top = active.nlargest(top_n, "predicted_uplift_pct")
    offers = [
        {
            "promo_type":           str(r.get("promo_type", "Discount")),
            "category":             str(r.get("category", "—")),
            "discount_pct":         round(float(r["disc_pct"]), 1),
            "target_segment":       str(r.get("target_segment", "All")),
            "on_target":            bool(r.get("on_target", False)),
            "predicted_uplift_pct": round(float(r["predicted_uplift_pct"]), 1),
        }
        for _, r in top.iterrows()
    ]
    return {
        "segment":            segment,
        "segment_size":       seg_size,
        "preferred_category": pref_cat,
        "preferred_channel":  pref_ch,
        "offers":             offers,
    }



@router.get("/next-best-offer")
def next_best_offer(customer_id: str = Query(..., description="Customer ID")):
    """Get the single best promotional offer for a customer right now."""
    result = get_next_best_offer(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/communication-timing")
def communication_timing(customer_id: str = Query(...)):
    """Predict the best send time and channel for customer communications."""
    result = get_communication_timing(customer_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/customers")
def list_customers(limit: int = Query(50, le=500)):
    """List available customer IDs for UI dropdowns."""
    from backend.utils.data_loader import get_customers
    df = get_customers()
    cols = [c for c in ["customer_id", "name", "segment", "city", "loyalty_tier",
                         "preferred_category", "preferred_channel"] if c in df.columns]
    return df[cols].head(limit).to_dict(orient="records")


@router.get("/segmentation")
def segmentation():
    """Return all customer segments for UI dropdowns."""
    from backend.utils.data_loader import get_customers
    df = get_customers()
    segs = sorted(df["segment"].dropna().unique().tolist()) if "segment" in df.columns else []
    return {"segments": [{"segment": s} for s in segs]}
