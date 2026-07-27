"""
Module 2 — Merchandising Intelligence Router
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.services.merchandising import (
    assortment_plan, forecast_demand, get_dynamic_price,
    optimise_promotion, monitor_competitor_prices,
)

router = APIRouter(prefix="/api/v1/merchandising", tags=["Merchandising Intelligence"])


@router.get("/assortment-plan")
def get_assortment_plan(region: Optional[str] = Query(None)):
    """Regional SKU assortment analysis — adds, drops and Pareto concentration."""
    return assortment_plan(region=region)



class PricingRequest(BaseModel):
    product_id: str
    store_id: Optional[str] = None


@router.get("/demand-forecast")
def demand_forecast(
    product_id: str = Query(...),
    days: int = Query(30, ge=7, le=90),
):
    """Forecast daily demand for a product for the next N days."""
    return forecast_demand(product_id, days)


@router.post("/dynamic-pricing")
def dynamic_pricing(req: PricingRequest):
    """Recommend optimal price for a product based on inventory, competition, and demand."""
    result = get_dynamic_price(req.product_id, req.store_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/promotion-optimization")
def promotion_optimization(promo_id: str = Query(...)):
    """Evaluate promotion uplift and cannibalization."""
    result = optimise_promotion(promo_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/competitor-monitoring")
def competitor_monitoring(
    product_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
):
    """Monitor competitor prices and generate alerts for significant gaps."""
    return {"results": monitor_competitor_prices(product_id, category)}


@router.get("/products")
def list_products(limit: int = Query(20, le=200), category: Optional[str] = None):
    """List products for UI dropdowns."""
    from backend.utils.data_loader import get_products
    df = get_products()
    if category:
        df = df[df["category"] == category]
    return df[["product_id","product_name","category","brand","price"]].head(limit).to_dict(orient="records")


@router.get("/promotions")
def list_promotions(limit: int = Query(50, le=500)):
    """List promotions for UI dropdowns."""
    from backend.utils.data_loader import get_promotions
    df = get_promotions()
    return df[["promo_id","promo_type","discount_pct","target_segment","is_active"]].head(limit).to_dict(orient="records")
