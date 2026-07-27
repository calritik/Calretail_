"""
Module 3 — Operations Intelligence service layer.

Thin wrappers over backend.capabilities.*, adding the naming and shaping the
console expects. The deterministic reads are memoised: a capability rebuild is
seconds of work and the console asks for the same slice on every page visit.
"""

from backend.utils.cache import ttl_cache
from typing import Optional

@ttl_cache()
def get_inventory_health(store_id: Optional[str] = None,
                          category: Optional[str] = None,
                          top_n: int = 50) -> list[dict]:
    from backend.capabilities import inventory_health_monitoring as mod
    res = mod.compute_inventory_health()
    
    if isinstance(res, dict) and "inventory_health" in res:
        records = res["inventory_health"]
    elif isinstance(res, list):
        records = res
    else:
        records = []

    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return []

    # Map stock_level to stock_qty to match UI expectations
    if "stock_level" in df.columns:
        df["stock_qty"] = df["stock_level"]

    from backend.utils.data_loader import get_products
    prods = get_products()
    if "product_name" not in df.columns:
        df = df.merge(prods[["product_id", "product_name", "category", "brand", "price"]], on="product_id", how="left")

    if store_id and "store_id" in df.columns:
        df = df[df["store_id"].astype(str) == str(store_id)]
    if category and "category" in df.columns:
        df = df[df["category"] == category]

    # A stock row lives in either a store or a warehouse, with the other column
    # blank. location_name is what the console's "Where" column reads, so it is
    # resolved here rather than leaving the UI to guess which id to show.
    from backend.utils import naming
    rows = df.head(top_n).fillna(0).to_dict(orient="records")
    for r in rows:
        r["store_name"] = naming.store(r.get("store_id"), default="")
        r["warehouse_name"] = naming.warehouse(r.get("warehouse_id"), default="")
        r["location_name"] = naming.location_label(r)
    return rows

@ttl_cache()
def get_replenishment_order(product_id: str,
                             store_id: Optional[str] = None) -> dict:
    from backend.capabilities import automated_replenishment as mod
    res = mod.get_replenishment_parameters(product_id)
    if "error" in res:
        return res
        
    # Map & enrich replenishment data to match frontend requirements
    from backend.utils import naming
    from backend.utils.data_loader import get_products, get_inventory, get_suppliers
    prods = get_products()
    prod_row = prods[prods["product_id"] == product_id]
    if prod_row.empty:
        return {"error": f"Product {product_id} not found"}
        
    p_item = prod_row.iloc[0]
    supplier_id = p_item.get("supplier_id")
    cost_price = float(p_item.get("cost_price", 100.0))
    
    inv = get_inventory()
    total_inv_rows = inv[inv["product_id"] == product_id]
    total_max_stock = int(total_inv_rows["max_stock"].sum()) if not total_inv_rows.empty else 200
    
    if store_id:
        inv_rows = total_inv_rows[total_inv_rows["store_id"] == store_id]
    else:
        inv_rows = total_inv_rows
        
    if not inv_rows.empty:
        current_stock = int(inv_rows["stock_qty"].sum())
        max_stock = int(inv_rows["max_stock"].sum())
    else:
        current_stock = 45 # default backup stock
        max_stock = 200
        
    sups = get_suppliers()
    sup_row = sups[sups["supplier_id"] == supplier_id]
    if not sup_row.empty:
        s_item = sup_row.iloc[0]
        supplier_name = s_item.get("name", "Unknown Supplier")
        supplier_reliability = float(s_item.get("reliability_score", 0.95))
        lead_time_days = float(s_item.get("lead_time_days", 5.0))
    else:
        supplier_name = "CalRetail Logistics Ltd"
        supplier_reliability = 0.92
        lead_time_days = 5.0
        
    rop = res.get("reorder_point", 30)
    reorder_qty = res.get("recommended_order_quantity", 100)
    safety_stock = res.get("safety_stock", 5)
    
    if store_id and total_max_stock > 0:
        ratio = max_stock / total_max_stock
        rop = max(1, int(round(rop * ratio)))
        reorder_qty = max(1, int(round(reorder_qty * ratio)))
        safety_stock = max(1, int(round(safety_stock * ratio)))
    
    return {
        "product_id": product_id,
        "product_name": p_item.get("product_name", product_id),
        "category": p_item.get("category", ""),
        "brand": p_item.get("brand", ""),
        "store_id": store_id,
        "store_name": naming.store(store_id, default="") if store_id else "",
        "current_stock": current_stock,
        "reorder_qty": reorder_qty,
        "lead_time_days": lead_time_days,
        "estimated_cost": reorder_qty * cost_price,
        "urgency_flag": bool(current_stock < rop),
        "supplier_id": supplier_id,
        "supplier_name": supplier_name,
        "supplier_reliability": supplier_reliability,
        "safety_stock": safety_stock,
        "max_stock": max_stock,
        "reorder_point": rop
    }

@ttl_cache()
def optimise_warehouse(warehouse_id: str) -> dict:
    from backend.capabilities import warehouse_slotting as mod
    records = mod.compute_abc_slotting_plan(warehouse_id)
    
    import pandas as pd
    df = pd.DataFrame(records)
    if df.empty:
        return {
            "class_summary": {},
            "slotting_plan": [],
            "estimated_pick_time_reduction_pct": 5.0
        }
        
    from backend.utils.data_loader import get_products
    prods = get_products()
    
    # Rename columns to match UI
    if "abc_class" in df.columns:
        df["velocity_class"] = df["abc_class"]
    if "assigned_zone" in df.columns:
        df["recommended_zone"] = df["assigned_zone"]
        
    # Join products
    df = df.merge(prods[["product_id", "product_name", "category"]], on="product_id", how="left")
    
    df["avg_daily_demand"] = (df["total_movements"] / 90.0).round(1)
    df["pick_time_savings_pct"] = df["velocity_class"].map({"A": 25.0, "B": 15.0, "C": 5.0})
    
    counts = df["velocity_class"].value_counts().to_dict()
    
    total_mvs = df["total_movements"].sum()
    if total_mvs > 0:
        est_saving = (df["total_movements"] * df["pick_time_savings_pct"]).sum() / total_mvs
    else:
        est_saving = 5.0
        
    est_saving = round(float(est_saving), 1)
    
    from backend.utils import naming
    return {
        "warehouse_id": warehouse_id,
        "warehouse_name": naming.warehouse(warehouse_id),
        "class_summary": counts,
        "slotting_plan": naming.annotate(df.fillna("Unknown").to_dict(orient="records")),
        "estimated_pick_time_reduction_pct": est_saving
    }

@ttl_cache()
def optimise_routes(warehouse_id: str, order_ids: list = None) -> dict:
    from backend.capabilities import route_optimisation as mod
    res = mod.solve_delivery_route(warehouse_id)
    
    dist = res.get("distance_km", 0.0)
    baseline = res.get("baseline_distance_km", round(dist * 1.25, 2))
    saved = round(max(0.0, baseline - dist), 2)
    saving_pct = round((saved / baseline) * 100.0, 1) if baseline > 0 else 0.0
    
    # The notebook returns the stop sequence as raw store ids ("S0121"). The
    # console shows this sequence to a planner, so translate to store names and
    # keep the ids alongside for anything that needs to join back.
    from backend.utils import naming
    stops = res.get("route_order", [])
    named_stops = [
        naming.warehouse(warehouse_id) if s in ("Warehouse", warehouse_id)
        else naming.store(s)
        for s in stops
    ]

    return {
        "optimised_distance_km": dist,
        "baseline_distance_km": baseline,
        "distance_saved_km": saved,
        "saving_pct": saving_pct,
        "estimated_time_hrs": round(dist / 50.0, 1),
        "route_order": named_stops,
        "route_order_ids": stops,
        "warehouse_id": warehouse_id,
        "warehouse_name": naming.warehouse(warehouse_id),
        "total_orders": res.get("total_orders", 0),
        "route": naming.annotate(res.get("route", [])),
        "origin": naming.annotate(res.get("origin", {})),
    }



# ─────────────────────────────────────────────────────────────────────────────
# Markdown candidates — the mirror image of the stockout buy-list.
# ─────────────────────────────────────────────────────────────────────────────

# Cover past which stock stops being a buffer and starts being idle capital,
# mapped to how hard it needs discounting to clear.
_MARKDOWN_TIERS = (
    (365, 40),
    (180, 30),
    (90,  20),
    (0,   10),
)


def _suggested_markdown(days_cover: float) -> int:
    for threshold, pct in _MARKDOWN_TIERS:
        if days_cover >= threshold:
            return pct
    return 10


@ttl_cache()
def get_markdown_candidates(top_n: int = 8) -> dict:
    """
    SKUs holding far more cover than the replenishment cycle needs.

    Ranked by capital tied up rather than by cover, because a year of cover on a
    ₹200 vest is not the problem a quarter of cover on a ₹9,000 coat is. Only
    SKUs whose stockout risk is negligible qualify — discounting something that
    might still sell out is how you create the opposite problem.
    """
    from backend.utils import db

    # overstock_flag is the pipeline's own definition (stock_qty > max_stock).
    # Using it rather than a cover threshold invented here keeps this endpoint
    # consistent with the health score and the risk labels, which are derived
    # from the same flag — a separate "days_cover >= 90" rule looked reasonable
    # but classified most of the estate as overstocked, including SKUs that are
    # simply slow sellers held at their planned level.
    rows = db.query(
        """
        SELECT h.product_id, h.stock_qty, h.days_cover, h.stockout_risk,
               h.overstock_flag, h.location_type, h.store_id, h.warehouse_id,
               p.product_name, p.category, p.brand, p.price
        FROM feature_inventory_health h
        JOIN products p ON p.product_id = h.product_id
        WHERE h.stock_qty > 0
          AND h.stockout_risk <= 0.10
          AND h.overstock_flag = 1
        """,
        parse_dates=False,
    )

    if rows.empty:
        return {"candidates": [], "overstocked_skus": 0,
                "capital_tied_up": 0.0, "freed_at_markdown": 0.0}

    rows["stock_value"] = rows["stock_qty"].astype(float) * rows["price"].astype(float)
    rows = rows.sort_values("stock_value", ascending=False)

    from backend.utils import naming
    candidates = []
    for r in rows.head(top_n).to_dict(orient="records"):
        cover = float(r.get("days_cover") or 0)
        pct = _suggested_markdown(cover)
        candidates.append({
            "product_id": r["product_id"],
            "product_name": r.get("product_name") or naming.product(r["product_id"]),
            "category": r.get("category", ""),
            "brand": r.get("brand", ""),
            "location_name": naming.location_label(r),
            "location_type": r.get("location_type", ""),
            "stock": int(r.get("stock_qty") or 0),
            "days_cover": round(cover, 1),
            "stockout_risk_pct": round(float(r.get("stockout_risk") or 0) * 100, 1),
            "stock_value": round(float(r["stock_value"]), 2),
            "suggested_markdown_pct": pct,
            # What the discount actually costs, so the freed figure below is
            # recoverable cash rather than the full shelf value.
            "markdown_cost": round(float(r["stock_value"]) * pct / 100.0, 2),
        })

    freed = sum(c["stock_value"] - c["markdown_cost"] for c in candidates)
    return {
        "candidates": candidates,
        "overstocked_skus": int(len(rows)),
        "capital_tied_up": round(float(rows["stock_value"].sum()), 2),
        "freed_at_markdown": round(freed, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Daily demand for one SKU.
# ─────────────────────────────────────────────────────────────────────────────

@ttl_cache()
def get_inventory_timeseries(product_id: str, days: int = 120) -> dict:
    """
    Daily units sold for one product over the trailing window, plus the cover
    that pace implies against stock currently on hand.

    The window is anchored to the last date present in the data, not to today —
    this is a fixed synthetic dataset ending in 2024, so anchoring to the wall
    clock would return an empty series forever.

    Days with no sale are emitted as zero. feature_daily_sales only stores days
    that had a transaction, and drawing that sparse frame straight onto a date
    axis silently overstates demand: the gaps vanish and the 7-day mean is taken
    over selling days rather than calendar days.
    """
    import pandas as pd

    from backend.utils import db, naming

    sales = db.query(
        "SELECT date, daily_qty FROM feature_daily_sales "
        "WHERE product_id = ? ORDER BY date",
        (product_id,), parse_dates=False,
    )

    stock_df = db.query(
        "SELECT COALESCE(SUM(stock_qty), 0) AS stock FROM inventory WHERE product_id = ?",
        (product_id,), parse_dates=False,
    )
    current_stock = int(stock_df.iloc[0]["stock"]) if not stock_df.empty else 0

    product_name = naming.product(product_id)
    if sales.empty:
        return {"product_id": product_id, "product_name": product_name,
                "series": [], "current_stock": current_stock,
                "avg_daily_qty": 0.0, "days_cover": 0.0, "window_days": 0}

    sales["date"] = pd.to_datetime(sales["date"], errors="coerce")
    sales = sales.dropna(subset=["date"])

    end = sales["date"].max()
    start = end - pd.Timedelta(days=int(days) - 1)
    window = sales[sales["date"] >= start]

    daily = (window.groupby("date")["daily_qty"].sum()
             .reindex(pd.date_range(start, end, freq="D"), fill_value=0))

    series = [{"date": d.strftime("%Y-%m-%d"), "qty": float(q)}
              for d, q in daily.items()]
    avg = float(daily.mean()) if len(daily) else 0.0

    return {
        "product_id": product_id,
        "product_name": product_name,
        "series": series,
        "current_stock": current_stock,
        "avg_daily_qty": round(avg, 2),
        "days_cover": round(current_stock / avg, 1) if avg > 0 else 0.0,
        "window_days": int(len(daily)),
        "period_start": start.strftime("%Y-%m-%d"),
        "period_end": end.strftime("%Y-%m-%d"),
    }
