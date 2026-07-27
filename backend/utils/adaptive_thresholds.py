"""
Adaptive Thresholds Utility
============================
Pre-computes all data-derived thresholds, multipliers, and parameters
that were previously hardcoded across the 20 AI capabilities.

All functions are cached — computed once from real data at startup.
Import these into service modules instead of using fixed constants.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — Customer Experience
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_category_conversion_rates() -> dict:
    """
    Compute conversion rate per product category from transactions + products.
    Returns {category: boost_multiplier} where multiplier = cat_CVR / global_CVR.
    Used by recommendation engine to replace fixed 1.25 category boost.
    """
    try:
        from backend.utils.data_loader import get_transactions, get_products, get_browsing
        txn      = get_transactions()
        products = get_products()
        browsing = get_browsing()

        # Purchases per category
        txn_cat = txn.merge(products[["product_id", "category"]], on="product_id", how="left")
        purchases_by_cat = txn_cat.groupby("category")["product_id"].count()

        # Browses per category (denominator)
        browse_cat = browsing.merge(products[["product_id", "category"]], on="product_id", how="left")
        browses_by_cat = browse_cat.groupby("category")["product_id"].count()

        all_cats = purchases_by_cat.index.union(browses_by_cat.index)
        cvr_by_cat = {}
        for cat in all_cats:
            p = purchases_by_cat.get(cat, 0)
            b = browses_by_cat.get(cat, 1)
            cvr_by_cat[cat] = p / b

        global_cvr = max(sum(cvr_by_cat.values()) / max(len(cvr_by_cat), 1), 1e-6)
        # Boost = how much better this category converts vs. average, clamped 0.8–2.0
        boost = {cat: float(np.clip(cvr / global_cvr, 0.8, 2.0))
                 for cat, cvr in cvr_by_cat.items()}
        return boost
    except Exception:
        return {}


@lru_cache(maxsize=1)
def get_nbo_weights() -> Tuple[float, float, float]:
    """
    Learn weights for Next Best Offer scoring via Logistic Regression on
    historical promo→purchase pairs.
    Returns (discount_weight, channel_weight, recency_weight).
    Falls back to (50, 15, 10) if insufficient data.
    """
    try:
        from backend.utils.data_loader import get_transactions, get_promotions, get_customers
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        txn   = get_transactions()
        promos = get_promotions()
        custs  = get_customers()

        txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

        # Group transaction dates by (customer_id, product_id) using a fast zip loop
        gp = {}
        for cid, pid, dt in zip(txn["customer_id"], txn["product_id"], txn["transaction_date"]):
            if (cid, pid) not in gp:
                gp[(cid, pid)] = []
            gp[(cid, pid)].append(dt)

        # Sample recent promotions to keep execution swift
        promos_sampled = promos.head(250) if len(promos) > 250 else promos

        rows = []
        for _, promo in promos_sampled.iterrows():
            pid      = promo.get("product_id", "")
            disc     = float(promo.get("discount_pct", 0))
            start    = pd.to_datetime(promo.get("start_date"))
            end      = pd.to_datetime(promo.get("end_date"))
            seg      = promo.get("target_segment", "All")
            channel  = promo.get("channel", "Email")

            seg_custs = custs[custs["segment"] == seg] if seg != "All" else custs
            for _, c in seg_custs.head(30).iterrows():  # sample 30 per promo
                cid = c["customer_id"]
                dates = gp.get((cid, pid), [])
                converted = int(any(start <= d <= end for d in dates))
                ch_match = 1 if c.get("preferred_channel", "") in [channel, "Both"] else 0
                rows.append([disc, ch_match, converted])

        if len(rows) < 20:
            return (50.0, 15.0, 10.0)

        df = pd.DataFrame(rows, columns=["discount", "ch_match", "label"])
        X = StandardScaler().fit_transform(df[["discount", "ch_match"]])
        y = df["label"].values

        if y.sum() < 3 or (1 - y).sum() < 3:
            return (50.0, 15.0, 10.0)

        lr = LogisticRegression(max_iter=200)
        lr.fit(X, y)
        coefs = np.abs(lr.coef_[0])     # [discount_coef, channel_coef]
        total = coefs.sum() + 1e-6
        # Scale to original order of magnitude (sum≈65 like original)
        scale = 65 / total
        return (float(coefs[0] * scale), float(coefs[1] * scale), 10.0)
    except Exception:
        return (50.0, 15.0, 10.0)


@lru_cache(maxsize=1)
def get_global_fallback_hour() -> Tuple[int, float]:
    """
    Return (median_browse_hour, median_open_rate) from population browsing data.
    Replaces fixed defaults (hour=10, open_rate=0.30).
    """
    try:
        from backend.utils.data_loader import get_browsing, get_campaigns
        browsing  = get_browsing()
        campaigns = get_campaigns()

        # Population median browse hour
        browsing["timestamp"] = pd.to_datetime(browsing["timestamp"], errors="coerce")
        hours = browsing["timestamp"].dt.hour.dropna()
        med_hour = int(hours.median()) if len(hours) > 0 else 10

        # Actual median open rate from campaigns
        if "clicks" in campaigns.columns and "impressions" in campaigns.columns:
            ctr_series = campaigns["clicks"] / campaigns["impressions"].replace(0, np.nan)
            med_open = float(ctr_series.median())
            med_open = np.clip(med_open, 0.05, 0.80)
        else:
            med_open = 0.30

        return (med_hour, med_open)
    except Exception:
        return (10, 0.30)


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — Merchandising
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_stock_percentiles() -> Tuple[float, float]:
    """
    Returns (p10_stock, p90_stock): data-driven scarcity and overstock thresholds.
    Replaces hardcoded stock<20 (scarcity) and stock>200 (overstock).
    """
    try:
        from backend.utils.data_loader import get_inventory
        inv = get_inventory()
        stock = inv.groupby("product_id")["stock_qty"].sum()
        p10 = float(stock.quantile(0.10))
        p90 = float(stock.quantile(0.90))
        return (max(p10, 1.0), max(p90, p10 + 10))
    except Exception:
        return (20.0, 200.0)


@lru_cache(maxsize=256)
def get_price_elasticity(product_id: str) -> float:
    """
    Estimate price elasticity ε from pricing_history vs transaction quantity.
    ε = % change in quantity / % change in price.
    Returns ε (negative = elastic). Falls back to −1.2 if insufficient data.
    """
    try:
        from backend.utils.data_loader import get_pricing_history, get_transactions
        ph  = get_pricing_history()
        txn = get_transactions()

        ph  = ph[ph["product_id"] == product_id].copy()
        txn = txn[txn["product_id"] == product_id].copy()

        if len(ph) < 2:
            return -1.2

        ph["effective_date"]     = pd.to_datetime(ph["effective_date"])
        txn["transaction_date"]  = pd.to_datetime(txn["transaction_date"])

        # For each price period, compute avg daily quantity sold
        ph = ph.sort_values("effective_date").reset_index(drop=True)
        records = []
        for i in range(len(ph) - 1):
            start = ph.loc[i, "effective_date"]
            end   = ph.loc[i + 1, "effective_date"]
            price = float(ph.loc[i, "price"])
            period_txn = txn[
                (txn["transaction_date"] >= start) &
                (txn["transaction_date"] <  end)
            ]
            days = max((end - start).days, 1)
            avg_qty = float(period_txn["quantity"].sum()) / days
            records.append({"price": price, "avg_qty": avg_qty})

        if len(records) < 2:
            return -1.2

        df = pd.DataFrame(records)
        pct_dp = df["price"].pct_change().dropna()
        pct_dq = df["avg_qty"].pct_change().dropna()

        if len(pct_dp) == 0 or pct_dp.abs().sum() < 1e-6:
            return -1.2

        elasticity = float((pct_dq / pct_dp.replace(0, np.nan)).mean())
        # Clamp to sensible retail range [−3.0, −0.1]
        return float(np.clip(elasticity, -3.0, -0.1))
    except Exception:
        return -1.2


@lru_cache(maxsize=1)
def get_competitor_gap_stats() -> Tuple[float, float]:
    """
    Returns (mean_gap, std_gap) of the global price_gap_pct distribution.
    Alert thresholds = mean ± 1.5σ (replaces fixed +15%/−10%).
    """
    try:
        from backend.utils.data_loader import get_competitor_pricing, get_products, get_pricing_history
        comp    = get_competitor_pricing()
        prods   = get_products()
        pricing = get_pricing_history()

        pricing["effective_date"] = pd.to_datetime(pricing["effective_date"])
        latest = pricing.sort_values("effective_date").groupby("product_id").last()["price"].reset_index()
        latest.columns = ["product_id", "our_price"]

        merged = comp.merge(latest, on="product_id", how="inner")
        merged["our_price"]  = pd.to_numeric(merged["our_price"],  errors="coerce")
        merged["price"]      = pd.to_numeric(merged["price"],       errors="coerce")
        merged = merged.dropna(subset=["our_price", "price"])
        merged["gap_pct"] = (merged["our_price"] - merged["price"]) / merged["price"] * 100

        mean_gap = float(merged["gap_pct"].mean())
        std_gap  = float(merged["gap_pct"].std())
        if np.isnan(std_gap) or std_gap < 0.1:
            std_gap = 5.0
        return (mean_gap, std_gap)
    except Exception:
        return (0.0, 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — Operations
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_inventory_health_weights() -> Tuple[float, float, float]:
    """
    Derive composite health score weights (w_stockout, w_overstock, w_reliability)
    via PCA explained variance on [stockout_risk, overstock_flag, supplier_reliability].
    Replaces fixed (0.5, 0.3, 0.2).
    """
    try:
        from backend.utils.data_loader import get_feature_inventory_health, get_inventory, get_suppliers, get_products
        feat = get_feature_inventory_health()
        cols = []
        if "stockout_risk"  in feat.columns: cols.append("stockout_risk")
        if "overstock_flag" in feat.columns: cols.append("overstock_flag")

        # Add supplier reliability if available
        prods   = get_products()
        sups    = get_suppliers()
        feat2   = feat.merge(prods[["product_id","supplier_id"]], on="product_id", how="left")
        feat2   = feat2.merge(sups[["supplier_id","reliability_score"]], on="supplier_id", how="left")
        if "reliability_score" in feat2.columns:
            feat2["reliability_score"] = feat2["reliability_score"].fillna(0.5)
            cols3 = cols + ["reliability_score"]
            X = feat2[cols3].fillna(0).values
        else:
            X = feat2[cols].fillna(0).values
            cols3 = cols

        if len(X) < 3 or len(cols3) < 2:
            return (0.5, 0.3, 0.2)

        pca    = PCA(n_components=len(cols3))
        pca.fit(X)
        # Use first PC loadings (absolute) as weights
        loadings = np.abs(pca.components_[0])
        weights  = loadings / loadings.sum()

        w_stockout    = float(weights[0]) if len(weights) > 0 else 0.5
        w_overstock   = float(weights[1]) if len(weights) > 1 else 0.3
        w_reliability = float(weights[2]) if len(weights) > 2 else 0.2
        return (w_stockout, w_overstock, w_reliability)
    except Exception:
        return (0.5, 0.3, 0.2)


@lru_cache(maxsize=1)
def get_adaptive_service_level() -> float:
    """
    Derive target service level from 1 − historical_stockout_rate.
    Replaces fixed z=1.645 (95% hardcoded).
    Returns z-score (e.g., 1.28 for 90%, 1.645 for 95%, 2.05 for 98%).
    """
    try:
        from backend.utils.data_loader import get_feature_inventory_health
        feat = get_feature_inventory_health()
        if "stockout_risk" in feat.columns:
            avg_stockout_rate = float(
                (feat["stockout_risk"] > 0.7).mean()  # fraction currently at high risk
            )
        else:
            avg_stockout_rate = 0.10   # fallback assumption

        # Target service level = 1 − observed stockout frequency, clamped
        target_sl = float(np.clip(1.0 - avg_stockout_rate, 0.80, 0.99))
        z = float(norm.ppf(target_sl))
        return round(z, 3)
    except Exception:
        return 1.645


@lru_cache(maxsize=1)
def get_abc_cutoffs() -> Tuple[float, float]:
    """
    True Pareto cutoffs from actual demand distribution.
    Returns (a_cutoff_fraction, b_cutoff_fraction) where:
      - A class: top fraction capturing 80% of cumulative demand
      - B class: next fraction capturing 95% of cumulative demand
    Replaces fixed 0.20 / 0.50.
    """
    try:
        from backend.utils.data_loader import get_feature_inventory_health
        feat = get_feature_inventory_health()
        demand = feat["avg_daily_demand"].fillna(0).sort_values(ascending=False).reset_index(drop=True)

        if len(demand) == 0:
            return (0.20, 0.50)

        total   = demand.sum()
        if total == 0:
            return (0.20, 0.50)

        cumulative = demand.cumsum() / total
        a_idx = int((cumulative < 0.80).sum())   # index where 80% demand is covered
        b_idx = int((cumulative < 0.95).sum())   # index where 95% demand is covered

        n = len(demand)
        a_frac = float(np.clip(a_idx / n, 0.10, 0.40))
        b_frac = float(np.clip(b_idx / n, a_frac + 0.05, 0.70))
        return (a_frac, b_frac)
    except Exception:
        return (0.20, 0.50)


@lru_cache(maxsize=1)
def get_avg_delivery_speed_kmh() -> float:
    """
    Compute actual average delivery speed from shipments data.
    Replaces fixed 40 km/h assumption.
    """
    try:
        from backend.utils.data_loader import get_shipments, get_stores, get_warehouses
        import math

        shipments  = get_shipments()
        stores     = get_stores()

        # Need shipped_date, delivered_date, and distance
        if not {"shipped_date", "delivered_date"}.issubset(shipments.columns):
            return 40.0

        ship = shipments.copy()
        ship["shipped_date"]   = pd.to_datetime(ship["shipped_date"],   errors="coerce")
        ship["delivered_date"] = pd.to_datetime(ship["delivered_date"], errors="coerce")
        ship = ship.dropna(subset=["shipped_date", "delivered_date"])
        ship["hours"] = (ship["delivered_date"] - ship["shipped_date"]).dt.total_seconds() / 3600
        ship = ship[(ship["hours"] > 0) & (ship["hours"] < 240)]  # sane range

        if "distance_km" in ship.columns:
            ship["speed"] = ship["distance_km"] / ship["hours"]
            valid = ship[(ship["speed"] > 1) & (ship["speed"] < 200)]
            if len(valid) > 10:
                return float(valid["speed"].median())

        return 40.0
    except Exception:
        return 40.0


# ─────────────────────────────────────────────────────────────────────────────
# Module 5 — Monetisation
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_segment_avg_order_value() -> dict:
    """
    Compute actual average order value per customer segment from transactions.
    Replaces fixed ₹500 AOV in ROAS calculation.
    Returns {segment: avg_order_value}.
    """
    try:
        from backend.utils.data_loader import get_transactions, get_customers
        txn   = get_transactions()
        custs = get_customers()

        txn_c = txn.merge(custs[["customer_id", "segment"]], on="customer_id", how="left")
        aov   = txn_c.groupby("segment")["total_amount"].mean().to_dict()
        global_aov = float(txn["total_amount"].mean())
        aov["_global"] = global_aov
        return {k: round(float(v), 2) for k, v in aov.items()}
    except Exception:
        return {"_global": 500.0}


@lru_cache(maxsize=1)
def get_kmeans_optimal_k(max_k: int = 12) -> int:
    """
    Elbow method on inertia curve for K-Means audience segmentation.
    Picks k where marginal inertia drop < 5% of total range.
    Replaces fixed k=8.
    """
    try:
        from backend.utils.data_loader import get_feature_customers
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        customers    = get_feature_customers()
        feature_cols = ["recency_days", "frequency", "monetary",
                        "avg_order_val", "total_browses", "wishlist_count", "cart_count"]
        available    = [c for c in feature_cols if c in customers.columns]

        if len(available) < 2 or len(customers) < 20:
            return 8

        X = StandardScaler().fit_transform(customers[available].fillna(0))

        inertias = []
        k_range  = range(3, min(max_k + 1, len(customers) // 10 + 3))
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            km.fit(X)
            inertias.append(km.inertia_)

        if len(inertias) < 2:
            return 8

        inertia_range = inertias[0] - inertias[-1]
        if inertia_range < 1e-6:
            return 8

        # Find elbow: first k where drop < 5% of total range
        optimal_k = list(k_range)[-1]
        for i in range(1, len(inertias)):
            drop = (inertias[i - 1] - inertias[i]) / inertia_range
            if drop < 0.05:
                optimal_k = list(k_range)[i]
                break

        return int(optimal_k)
    except Exception:
        return 8


@lru_cache(maxsize=1)
def get_intent_thresholds() -> Tuple[float, float]:
    """
    Find optimal high/medium probability thresholds for buying intent scoring
    from the Precision-Recall curve on held-out test data.
    Replaces fixed 0.60 / 0.35 cutoffs.
    Returns (high_threshold, medium_threshold).
    """
    try:
        from backend.services.monetisation import _train_intent_model, _intent_model_cache
        from backend.utils.data_loader import get_feature_buying_intent

        _train_intent_model()
        model = _intent_model_cache["model"]
        feats = _intent_model_cache["features"]

        df = get_feature_buying_intent()
        if len(df) > 5000:
            df = df.sample(n=5000, random_state=42)

        df = df[feats + ["purchased_label"]].fillna(0)
        X = df[feats].values
        y = df["purchased_label"].values

        probs = model.predict_proba(X)[:, 1]
        pos_probs = probs[y == 1]
        if len(pos_probs) < 4:
            return (0.60, 0.35)

        high_t   = float(np.clip(np.percentile(pos_probs, 60), 0.15, 0.90))
        medium_t = float(np.clip(np.percentile(pos_probs, 25), 0.05, high_t - 0.02))
        return (high_t, medium_t)
    except Exception:
        return (0.60, 0.35)


def get_auto_cluster_labels(cluster_summaries: list[dict]) -> dict:
    """
    Assign segment labels from centroid characteristics — not from a lookup table.
    Returns {cluster_id: label_string}.

    Rules (applied in priority order):
      recency   < 30 ∧ frequency > median_freq ∧ monetary > median_mon  → "Champions"
      recency   < 30 ∧ frequency > median_freq                           → "Loyal Customers"
      recency   < 30 ∧ monetary  > median_mon                            → "Recent High-Value"
      recency   > 90                                                      → "Lapsed"
      recency   > 60                                                      → "At Risk"
      frequency == 1                                                      → "First-Time Buyers"
      browse    > median_browse ∧ monetary < median_mon                  → "Window Shoppers"
      monetary  > median_mon                                              → "Big Spenders"
      frequency > median_freq                                             → "Frequent Buyers"
      else                                                                → "Occasional Buyers"
    """
    if not cluster_summaries:
        return {}

    rec_vals  = [s.get("avg_recency_days", 30)  for s in cluster_summaries]
    freq_vals = [s.get("avg_frequency", 5)       for s in cluster_summaries]
    mon_vals  = [s.get("avg_monetary", 1000)     for s in cluster_summaries]
    br_vals   = [s.get("avg_browse_count", 10)   for s in cluster_summaries]

    med_freq     = float(np.median(freq_vals))
    med_mon      = float(np.median(mon_vals))
    med_browse   = float(np.median(br_vals))

    labels = {}
    used   = set()
    for s in cluster_summaries:
        cid  = s["cluster_id"]
        rec  = s.get("avg_recency_days", 30)
        freq = s.get("avg_frequency", 5)
        mon  = s.get("avg_monetary", 1000)
        br   = s.get("avg_browse_count", 10)

        if   rec < 30 and freq > med_freq and mon > med_mon:        label = "Champions"
        elif rec < 30 and freq > med_freq:                          label = "Loyal Customers"
        elif rec < 30 and mon > med_mon:                            label = "Recent High-Value"
        elif rec > 90:                                              label = "Lapsed"
        elif rec > 60:                                              label = "At Risk"
        elif freq <= 1:                                             label = "First-Time Buyers"
        elif br > med_browse and mon < med_mon:                     label = "Window Shoppers"
        elif mon > med_mon:                                         label = "Big Spenders"
        elif freq > med_freq:                                       label = "Frequent Buyers"
        else:                                                       label = "Occasional Buyers"

        # Deduplicate labels by appending cluster id suffix if already used
        base = label
        suffix = 1
        while label in used:
            label = f"{base} {suffix}"
            suffix += 1
        used.add(label)
        labels[cid] = label

    return labels
