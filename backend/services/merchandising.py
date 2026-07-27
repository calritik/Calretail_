"""
Module 2 — Merchandising Intelligence service layer.

Thin wrappers over backend.capabilities.*, memoised because the underlying
database is read-only and the console re-requests the same slices.
"""

from backend.utils.cache import ttl_cache

@ttl_cache()
def forecast_demand(product_id: str, days: int = 7) -> dict:
    from backend.capabilities import demand_forecasting as mod
    res = mod.get_demand_forecast(product_id, forecast_days=days)

    if not isinstance(res, dict):
        res = {}

    import datetime
    import numpy as np

    forecast_list = res.get("forecast", [])
    historical_list = res.get("historical", [])

    # Anchor forecast dates the day *after* the product's real last sale date
    # (from the notebook's historical series) so the chart is continuous —
    # falls back to today only when there's no sales history at all.
    if historical_list:
        anchor = datetime.date.fromisoformat(historical_list[-1]["date"])
    else:
        anchor = datetime.date.today()

    mapped_forecast = []
    total_forecast = 0.0
    for item in forecast_list:
        day_idx = item.get("day", 1)
        qty = float(item.get("forecast_qty", 0.0))
        total_forecast += qty
        date_str = str(anchor + datetime.timedelta(days=day_idx))

        # Confidence band from the model's own real held-out error (MAPE),
        # not an arbitrary guess.
        mape_frac = float(res.get("mape", 15.0)) / 100.0
        std_err = max(mape_frac * qty, 0.5)
        mapped_forecast.append({
            "date": date_str,
            "predicted_qty": qty,
            "upper_bound": round(qty + 1.96 * std_err, 1),
            "lower_bound": round(max(0.0, qty - 1.96 * std_err), 1)
        })

    mape = res.get("mape", 15.0)
    if hasattr(mape, "item"):
        mape = mape.item()
    mape = float(mape)

    from backend.utils import naming
    from backend.utils.data_loader import product as product_row
    prod = product_row(product_id) or {}

    return {
        "product_id": product_id,
        "product_name": prod.get("product_name") or naming.product(product_id),
        "category": prod.get("category", ""),
        "brand": prod.get("brand", ""),
        "total_forecast": round(total_forecast, 1),
        "avg_daily_demand": round(total_forecast / max(1, days), 2),
        "mape_estimate": round(mape, 2),
        "model": res.get("model", "XGBoost (Global Multi-Product Regressor)"),
        "forecast": mapped_forecast,
        "historical": historical_list
    }

@ttl_cache()
def get_dynamic_price(product_id: str, store_id: str = None) -> dict:
    from backend.capabilities import dynamic_pricing as mod
    res = mod.recommend_dynamic_price(product_id)
    
    if not isinstance(res, dict):
        res = {}
    if "error" in res:
        return res
        
    current_price = float(res.get("current_price", 0.0))
    recommended_price = float(res.get("recommended_price", current_price))
    
    price_delta_pct = 0.0
    if current_price > 0.0:
        price_delta_pct = ((recommended_price - current_price) / current_price) * 100.0
        
    floor_price = current_price * 0.75
    rationale = res.get("inventory_nudge", "Stock level stable. Pricing aligned with competitor average.")
    avg_competitor_price = float(res.get("competitor_avg", current_price))
    
    # Calculate min competitor price dynamically
    try:
        cust_pr = mod.cust_pr
        comp_matches = cust_pr[cust_pr['product_id'] == product_id]
        if not comp_matches.empty:
            min_competitor_price = float(comp_matches['price'].min())
        else:
            min_competitor_price = avg_competitor_price * 0.92
    except Exception:
        min_competitor_price = avg_competitor_price * 0.92
        
    expected_revenue_lift_pct = float(res.get("est_revenue_lift_pct", 0.0))
    
    return {
        "product_id": product_id,
        "product_name": res.get("product_name", ""),
        "current_price": round(current_price, 2),
        "recommended_price": round(recommended_price, 2),
        "price_delta_pct": round(price_delta_pct, 2),
        "floor_price": round(floor_price, 2),
        "expected_revenue_lift_pct": round(expected_revenue_lift_pct, 2),
        "rationale": rationale,
        "avg_competitor_price": round(avg_competitor_price, 2),
        "min_competitor_price": round(min_competitor_price, 2),
        "stock_level": int(res.get("stock_level", 0))
    }

@ttl_cache()
def optimise_promotion(promo_id: str) -> dict:
    from backend.capabilities import promotion_optimization as mod
    res = mod.analyze_promo_performance(promo_id)
    
    if not isinstance(res, dict):
        res = {}
    if "error" in res:
        return res
        
    control_revenue = float(res.get("baseline_revenue", 0.0))
    treated_revenue = float(res.get("promo_revenue", 0.0))
    incremental_revenue = float(res.get("incremental_uplift_value", 0.0))
    
    # Compute uplift_pct which stream lit app metrics display
    uplift_pct = 0.0
    if control_revenue > 0.0:
        uplift_pct = ((treated_revenue - control_revenue) / control_revenue) * 100.0
        
    # Cannibalization rate (from percentage to fraction: cannibalization_rate_pct / 100)
    cannibalization_pct = float(res.get("cannibalization_rate_pct", 0.0))
    cannibalization_rate = cannibalization_pct / 100.0
    
    confidence = float(res.get("confidence_level", 0.92))
    
    # A promotion has no name column of its own, so its readable label is built
    # from what a merchandiser actually recognises it by: type, depth, audience.
    from backend.utils import db, naming
    pid = res.get("product_id", "")
    promo = db.read_table("promotions", where="promo_id = ?", params=(promo_id,), limit=1)
    if promo.empty:
        promo_label = promo_id
        promo_row = {}
    else:
        promo_row = promo.iloc[0].to_dict()
        promo_label = (f"{promo_row.get('promo_type', 'Promotion')} · "
                       f"{float(promo_row.get('discount_pct', 0) or 0) * 100:.0f}% off · "
                       f"{promo_row.get('target_segment', 'All shoppers')}")

    return {
        "promo_id": promo_id,
        "promo_name": promo_label,
        "promo_type": promo_row.get("promo_type", ""),
        "discount_pct": promo_row.get("discount_pct", 0),
        "target_segment": promo_row.get("target_segment", ""),
        "product_id": pid,
        "product_name": naming.product(pid, default="") if pid else "",
        "control_revenue": round(control_revenue, 2),
        "treated_revenue": round(treated_revenue, 2),
        "incremental_revenue": round(incremental_revenue, 2),
        "uplift_pct": round(uplift_pct, 2),
        "cannibalization_rate": round(cannibalization_rate, 4),
        "confidence": round(confidence, 2)
    }

@ttl_cache()
def monitor_competitor_prices(product_id: str = None, category: str = None) -> list[dict]:
    from backend.capabilities import competitor_price_monitoring as mod
    res = mod.detect_pricing_outliers()
    
    if not isinstance(res, list):
        if isinstance(res, dict) and "alerts" in res:
            res = res["alerts"]
        elif isinstance(res, dict) and "outliers" in res:
            res = res["outliers"]
        else:
            res = []
            
    # Get categories to populate "category"
    categories_map = {}
    try:
        prod_df = mod.prod
        categories_map = dict(zip(prod_df["product_id"], prod_df["category"]))
    except Exception:
        pass
        
    processed_results = []
    for item in res:
        pid = item.get("product_id")
        pcat = categories_map.get(pid, item.get("category", "General"))
        
        # Filter by product_id
        if product_id and pid != product_id:
            continue
            
        # Filter by category
        if category and pcat != category:
            continue
            
        our_price = float(item.get("our_price", 0.0))
        competitor_mean = float(item.get("competitor_mean", 0.0))
        gap_pct = float(item.get("gap_pct", 0.0))
        
        action = item.get("recommended_action", "Maintain current pricing")
        alert_flag = "Maintain" not in action
        
        processed_results.append({
            "product_id": pid,
            "product_name": item.get("product_name", ""),
            "category": pcat,
            "our_price": our_price,
            "avg_competitor_price": competitor_mean,
            "price_gap_pct": gap_pct,
            "z_score": float(item.get("z_score", 0.0)),
            "status": item.get("status", ""),
            "recommended_action": action,
            "alert_flag": alert_flag
        })
        
    return processed_results


@ttl_cache()
def assortment_plan(region: str = None) -> dict:
    """
    Data-driven assortment analysis:
    - Joins orders → stores → products to get regional SKU revenue
    - Computes 80/20 Pareto concentration per region
    - Identifies add candidates (proven elsewhere but underperforming here)
    - Identifies drop candidates (below median revenue, inventory potentially tied up)
    """
    import pandas as pd
    from backend.utils.data_loader import (
        get_orders, get_stores, get_products, get_inventory
    )

    orders  = get_orders()
    stores  = get_stores()
    prods   = get_products()
    inv     = get_inventory()

    # Only confirmed revenue orders
    if "status" in orders.columns:
        orders = orders[~orders["status"].isin(["Cancelled", "Returned"])]

    # Join store region
    if "store_id" in orders.columns and "store_id" in stores.columns:
        store_region = stores[["store_id", "region"]].drop_duplicates("store_id")
        orders = orders.merge(store_region, on="store_id", how="left")
    else:
        orders["region"] = "All"

    # Join product info
    prod_cols = ["product_id", "product_name", "category", "price"]
    prod_cols = [c for c in prod_cols if c in prods.columns]
    orders = orders.merge(prods[prod_cols].drop_duplicates("product_id"),
                          on="product_id", how="left")

    # Revenue column
    rev_col = "total_amount" if "total_amount" in orders.columns else \
              "price" if "price" in orders.columns else None
    if rev_col is None:
        orders["_rev"] = 1.0
        rev_col = "_rev"

    orders["_rev"] = orders[rev_col].astype(float).fillna(0)

    if region:
        orders = orders[orders["region"] == region]

    # Global stats
    catalogue_skus = int(prods["product_id"].nunique())
    skus_selling   = int(orders["product_id"].nunique())
    revenue_total  = float(orders["_rev"].sum())
    orders_analysed = int(len(orders))

    # Pareto — share of SKUs that earn the first 80% revenue
    sku_rev = orders.groupby("product_id")["_rev"].sum().sort_values(ascending=False)
    cumsum  = sku_rev.cumsum()
    pareto_skus = int((cumsum <= revenue_total * 0.80).sum())
    pareto_sku_pct = round(pareto_skus / max(skus_selling, 1) * 100, 1)

    # ── Per-region breakdown ──────────────────────────────────────────────────
    by_region = []
    all_regions = orders["region"].dropna().unique()
    for reg in sorted(all_regions):
        reg_ord = orders[orders["region"] == reg]
        reg_rev = float(reg_ord["_rev"].sum())
        reg_skus_selling = int(reg_ord["product_id"].nunique())

        # Pareto within region
        reg_sku_rev = reg_ord.groupby("product_id")["_rev"].sum().sort_values(ascending=False)
        reg_cum = reg_sku_rev.cumsum()
        reg_pareto = int((reg_cum <= reg_rev * 0.80).sum())
        reg_pareto_pct = round(reg_pareto / max(reg_skus_selling, 1) * 100, 1)

        # Threshold for add/drop within this region
        median_rev = float(reg_sku_rev.median()) if len(reg_sku_rev) else 0.0
        reg_products = set(reg_ord["product_id"].unique())

        # Drop: product sells here but below 25% of region median
        drop_ids = set(reg_sku_rev[reg_sku_rev < median_rev * 0.25].index.tolist())

        # Add: product sells well in OTHER regions (> peer median there) but not here
        other_sku_rev = orders[orders["region"] != reg].groupby("product_id")["_rev"].sum()
        other_median = float(other_sku_rev.median()) if len(other_sku_rev) else 0.0
        add_ids = set(other_sku_rev[
            (other_sku_rev > other_median) &
            (~other_sku_rev.index.isin(reg_products))
        ].head(15).index.tolist())

        by_region.append({
            "region":         reg,
            "skus_selling":   reg_skus_selling,
            "revenue":        round(reg_rev, 2),
            "pareto_sku_pct": reg_pareto_pct,
            "add":            len(add_ids),
            "drop":           len(drop_ids),
        })

    # ── Add / drop candidate details ─────────────────────────────────────────
    # Inventory index for tied capital
    inv_stock = {}
    if "product_id" in inv.columns and "quantity_on_hand" in inv.columns:
        inv_stock = dict(zip(inv["product_id"],
                             inv["quantity_on_hand"].astype(float).fillna(0)))

    prod_info = prods.set_index("product_id")[
        [c for c in ["product_name", "category", "price"] if c in prods.columns]
    ].to_dict("index")

    # Add candidates: top opportunities from other-region best sellers
    other_best = orders[orders["region"] != (region or "")]\
                     .groupby("product_id")["_rev"].sum()\
                     .sort_values(ascending=False).head(20)
    selling_everywhere = set(orders["product_id"].unique())
    add_candidates = []
    for pid, opp in other_best.items():
        if pid in selling_everywhere:
            continue
        info = prod_info.get(pid, {})
        add_candidates.append({
            "product_id":   pid,
            "product_name": info.get("product_name", pid),
            "category":     info.get("category", "—"),
            "region":       region or "New",
            "opportunity":  round(float(opp) * 0.15, 2),  # estimated share
            "status":       "Not stocked in target region",
        })
        if len(add_candidates) >= 10:
            break

    # Drop candidates: consistent lowest performers
    global_sku_rev = orders.groupby("product_id")["_rev"].sum()
    global_median  = float(global_sku_rev.median()) if len(global_sku_rev) else 0.0
    drop_ids_global = global_sku_rev[global_sku_rev < global_median * 0.25].index.tolist()
    drop_candidates = []
    for pid in drop_ids_global[:10]:
        info = prod_info.get(pid, {})
        stock = inv_stock.get(pid, 0.0)
        price = float(info.get("price", 0) or 0)
        drop_candidates.append({
            "product_id":   pid,
            "product_name": info.get("product_name", pid),
            "category":     info.get("category", "—"),
            "region":       region or "All",
            "tied_capital": round(stock * price, 2),
            "status":       "Below 25% of median SKU revenue",
        })

    return {
        "catalogue_skus":       catalogue_skus,
        "skus_selling":         skus_selling,
        "revenue_total":        round(revenue_total, 2),
        "orders_analysed":      orders_analysed,
        "pareto_sku_pct":       pareto_sku_pct,
        "add_candidates_total": len(add_candidates),
        "drop_candidates_total":len(drop_candidates),
        "opportunity_value":    round(sum(a["opportunity"] for a in add_candidates), 2),
        "tied_capital":         round(sum(d["tied_capital"] for d in drop_candidates), 2),
        "by_region":            by_region,
        "add_candidates":       add_candidates,
        "drop_candidates":      drop_candidates,
    }
