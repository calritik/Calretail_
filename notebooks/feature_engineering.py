"""
Nexalyze — Feature Engineering Script
Reads the cleaned tables from the shipped database and writes ML-ready
feature tables back into it as feature_* tables.
Run via: python -m notebooks.feature_engineering  OR  python -m notebooks.build_db
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from notebooks.pipeline_io import load_processed, save_processed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("nexalyze.features")


def load(name: str) -> pd.DataFrame:
    return load_processed(name)


def save(df: pd.DataFrame, name: str) -> None:
    save_processed(df, f"feature_{name}")
    log.info(f"Saved feature_{name}  ({len(df):,} rows, {df.shape[1]} cols)")


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOMER RFM + BEHAVIOURAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_customer_features():
    log.info("Building customer RFM features...")
    customers = load("customers")
    txn = load("transactions")
    browsing = load("browsing_history")
    wishlist = load("wishlist")
    cart = load("shopping_cart")
    reviews = load("customer_reviews")

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])
    snapshot_date = txn["transaction_date"].max() + pd.Timedelta(days=1)

    # ── RFM ──────────────────────────────────────────────────────────────────
    rfm = txn.groupby("customer_id").agg(
        recency_days  = ("transaction_date", lambda x: (snapshot_date - x.max()).days),
        frequency     = ("transaction_id", "count"),
        monetary      = ("total_amount", "sum"),
        avg_order_val = ("total_amount", "mean"),
        first_purchase= ("transaction_date", "min"),
        last_purchase = ("transaction_date", "max"),
    ).reset_index()

    # RFM scores (1-5)
    for col, asc in [("recency_days", False), ("frequency", True), ("monetary", True)]:
        score_col = col.split("_")[0] + "_score"
        rfm[score_col] = pd.qcut(rfm[col], q=5,
                                  labels=[1,2,3,4,5] if asc else [5,4,3,2,1],
                                  duplicates="drop").astype(float).fillna(3)
    rfm["rfm_score"] = rfm["recency_score"] + rfm["frequency_score"] + rfm["monetary_score"]
    rfm["customer_lifetime_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days

    # ── Browse / Wishlist / Cart stats ───────────────────────────────────────
    browse_agg = browsing.groupby("customer_id").agg(
        total_browses   = ("browse_id", "count"),
        avg_dwell_secs  = ("dwell_seconds", "mean"),
        cart_adds       = ("added_to_cart", "sum"),
        browse_to_buy   = ("purchased", "mean"),
    ).reset_index()

    wish_agg = wishlist.groupby("customer_id").agg(
        wishlist_count  = ("wishlist_id", "count"),
        wishlist_bought = ("is_purchased", "sum"),
    ).reset_index()

    cart_agg = cart.groupby("customer_id").agg(
        cart_count     = ("cart_id", "count"),
        cart_abandoned = ("status", lambda x: (x == "Abandoned").sum()),
    ).reset_index()

    review_agg = reviews.groupby("customer_id").agg(
        review_count    = ("review_id", "count"),
        avg_rating_given= ("rating", "mean"),
    ).reset_index()

    # ── Merge all ─────────────────────────────────────────────────────────────
    feat = customers.merge(rfm,        on="customer_id", how="left")
    feat = feat.merge(browse_agg,      on="customer_id", how="left")
    feat = feat.merge(wish_agg,        on="customer_id", how="left")
    feat = feat.merge(cart_agg,        on="customer_id", how="left")
    feat = feat.merge(review_agg,      on="customer_id", how="left")

    num_cols = ["recency_days","frequency","monetary","avg_order_val",
                "total_browses","avg_dwell_secs","browse_to_buy",
                "wishlist_count","cart_count"]
    feat[num_cols] = feat[num_cols].fillna(0)

    # Encode categoricals for ML
    le = LabelEncoder()
    for col in ["gender","income_bracket","segment","preferred_category","loyalty_tier"]:
        if col in feat.columns:
            feat[f"{col}_enc"] = le.fit_transform(feat[col].astype(str))

    save(feat, "customers")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRODUCT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_product_features():
    log.info("Building product features...")
    products = load("products")
    txn = load("transactions")
    reviews = load("customer_reviews")
    returns = load("returns")
    inventory = load("inventory")

    # Sales stats
    sales_agg = txn.groupby("product_id").agg(
        total_qty_sold  = ("quantity", "sum"),
        total_revenue   = ("total_amount", "sum"),
        avg_sale_price  = ("unit_price", "mean"),
        avg_discount    = ("discount_pct", "mean"),
        n_transactions  = ("transaction_id", "count"),
        online_share    = ("channel", lambda x: (x == "Online").mean()),
    ).reset_index()

    review_agg = reviews.groupby("product_id").agg(
        avg_rating      = ("rating", "mean"),
        review_count    = ("review_id", "count"),
        pct_positive    = ("sentiment_label", lambda x: (x == "Positive").mean() * 100),
        pct_negative    = ("sentiment_label", lambda x: (x == "Negative").mean() * 100),
    ).reset_index()

    return_agg = returns.groupby("product_id").agg(
        return_count    = ("return_id", "count"),
    ).reset_index()

    inv_agg = inventory.groupby("product_id").agg(
        total_stock     = ("stock_qty", "sum"),
        avg_days_on_hand= ("days_on_hand", "mean"),
        stockout_locations = ("stockout_risk", "sum"),
    ).reset_index()

    feat = products.merge(sales_agg,  on="product_id", how="left")
    feat = feat.merge(review_agg,     on="product_id", how="left")
    feat = feat.merge(return_agg,     on="product_id", how="left")
    feat = feat.merge(inv_agg,        on="product_id", how="left")

    feat["total_qty_sold"]  = feat["total_qty_sold"].fillna(0)
    feat["total_revenue"]   = feat["total_revenue"].fillna(0)
    feat["return_count"]    = feat["return_count"].fillna(0)
    feat["return_rate"]     = np.where(
        feat["total_qty_sold"] > 0,
        feat["return_count"] / feat["total_qty_sold"],
        0).round(4)
    feat["sell_through_rate"] = np.where(
        (feat["total_stock"] + feat["total_qty_sold"]) > 0,
        feat["total_qty_sold"] / (feat["total_stock"] + feat["total_qty_sold"]),
        0).round(4)

    # Velocity class: A = top 20%, B = next 30%, C = bottom 50%
    feat["velocity_class"] = pd.qcut(
        feat["total_qty_sold"].fillna(0),
        q=[0, 0.5, 0.8, 1.0],
        labels=["C", "B", "A"]
    ).astype(str)

    # Encode categoricals
    le = LabelEncoder()
    for col in ["category","sub_category","brand","season","style_tag","color"]:
        if col in feat.columns:
            feat[f"{col}_enc"] = le.fit_transform(feat[col].astype(str))

    save(feat, "products")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSACTION TIME-SERIES FEATURES (for forecasting)
# ─────────────────────────────────────────────────────────────────────────────

def build_transaction_features():
    log.info("Building transaction daily-aggregation features for forecasting...")
    txn = load("transactions")
    holidays = load("holiday_calendar")
    weather = load("weather")

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    # Daily sales at product × store level (for demand forecasting)
    daily = txn.groupby(["product_id","transaction_date"]).agg(
        daily_qty    = ("quantity", "sum"),
        daily_revenue= ("total_amount", "sum"),
        n_orders     = ("transaction_id", "count"),
    ).reset_index()

    # Holiday features
    holidays["date"] = pd.to_datetime(holidays["date"])
    daily = daily.merge(
        holidays[["date","is_holiday","is_sale_season","season","day_of_week","is_weekend"]],
        left_on="transaction_date", right_on="date", how="left")

    # Lag features (7-day and 14-day)
    daily = daily.sort_values(["product_id","transaction_date"])
    daily["lag_7"]  = daily.groupby("product_id")["daily_qty"].shift(7)
    daily["lag_14"] = daily.groupby("product_id")["daily_qty"].shift(14)
    daily["rolling_7_mean"]  = daily.groupby("product_id")["daily_qty"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    daily["rolling_30_mean"] = daily.groupby("product_id")["daily_qty"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean())

    daily["month"]       = daily["transaction_date"].dt.month
    daily["day_of_month"]= daily["transaction_date"].dt.day
    daily["week_number"] = daily["transaction_date"].dt.isocalendar().week.astype(int)
    daily["quarter"]     = daily["transaction_date"].dt.quarter

    save(daily, "daily_sales")
    return daily


# ─────────────────────────────────────────────────────────────────────────────
# 4. INVENTORY HEALTH FEATURES (for smart inventory + replenishment)
# ─────────────────────────────────────────────────────────────────────────────

def build_inventory_features():
    log.info("Building inventory health features...")
    inventory = load("inventory")
    txn = load("transactions")
    products = load("products")

    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    # Last 30-day demand per product-store
    recent = txn[txn["transaction_date"] >= txn["transaction_date"].max() - pd.Timedelta(days=30)]
    demand_30 = recent.groupby("product_id").agg(
        demand_30d = ("quantity", "sum")
    ).reset_index()
    demand_30["avg_daily_demand"] = (demand_30["demand_30d"] / 30).round(2)

    feat = inventory.merge(demand_30, on="product_id", how="left")
    feat["demand_30d"]       = feat["demand_30d"].fillna(1)
    feat["avg_daily_demand"] = feat["avg_daily_demand"].fillna(0.1)

    # Derived metrics
    feat["days_cover"]       = (feat["stock_qty"] / feat["avg_daily_demand"].clip(0.01)).round(1)
    feat["reorder_urgency"]  = (feat["stock_qty"] <= feat["reorder_point"]).astype(int)
    feat["overstock_flag"]   = (feat["stock_qty"] > feat["max_stock"]).astype(int)

    # Health score (0–100)
    feat["health_score"] = (
        np.clip(feat["days_cover"] / 30 * 60, 0, 60) +         # Cover days (max 60pts)
        (1 - feat["reorder_urgency"]) * 25 +                    # Not in reorder (25pts)
        (1 - feat["overstock_flag"]) * 15                       # Not overstocked (15pts)
    ).round(1)

    feat["risk_label"] = pd.cut(
        feat["health_score"], bins=[0, 30, 60, 100], labels=["High Risk","Medium","Healthy"]
    ).astype(str)

    save(feat, "inventory_health")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 5. BUYING INTENT FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_intent_features():
    log.info("Building buying intent features...")
    browsing = load("browsing_history")
    cart = load("shopping_cart")
    wishlist = load("wishlist")
    search = load("search_history")
    txn = load("transactions")

    browsing["timestamp"] = pd.to_datetime(browsing["timestamp"])
    txn["transaction_date"] = pd.to_datetime(txn["transaction_date"])

    # Session-level: aggregate per customer-product pair
    browse_agg = browsing.groupby(["customer_id","product_id"]).agg(
        total_views   = ("page_views", "sum"),
        total_dwell   = ("dwell_seconds", "sum"),
        sessions      = ("browse_id", "count"),
        last_browse   = ("timestamp", "max"),
    ).reset_index()

    cart_flag = cart[cart["status"].isin(["Active","Abandoned"])][["customer_id","product_id"]].copy()
    cart_flag["cart_flag"] = 1

    wish_flag = wishlist[["customer_id","product_id"]].copy()
    wish_flag["wishlist_flag"] = 1

    # Was it purchased within 7 days of browsing?
    # (Simplified: if customer purchased same product, label = 1)
    purchased = txn[["customer_id","product_id"]].drop_duplicates()
    purchased["purchased_label"] = 1

    feat = browse_agg.merge(cart_flag,   on=["customer_id","product_id"], how="left")
    feat = feat.merge(wish_flag,         on=["customer_id","product_id"], how="left")
    feat = feat.merge(purchased,         on=["customer_id","product_id"], how="left")

    feat["cart_flag"]       = feat["cart_flag"].fillna(0).astype(int)
    feat["wishlist_flag"]   = feat["wishlist_flag"].fillna(0).astype(int)
    feat["purchased_label"] = feat["purchased_label"].fillna(0).astype(int)

    # Intent score (rule-based for interpretability)
    feat["intent_score"] = (
        feat["total_views"].clip(0, 20) / 20 * 30 +      # views (max 30)
        feat["cart_flag"] * 35 +                          # in cart (35)
        feat["wishlist_flag"] * 20 +                      # wishlisted (20)
        feat["total_dwell"].clip(0, 300) / 300 * 15       # dwell time (max 15)
    ).round(2)

    feat["intent_label"] = pd.cut(
        feat["intent_score"],
        bins=[0, 30, 60, 100],
        labels=["Low","Medium","High"]
    ).astype(str)

    save(feat, "buying_intent")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUPPORT TICKET FEATURES (for triage)
# ─────────────────────────────────────────────────────────────────────────────

def build_ticket_features():
    log.info("Building ticket triage features...")
    tickets = load("support_tickets")
    customers = load("customers")

    # Text length feature
    tickets["description"] = tickets["description"].fillna("").astype(str)
    tickets["text_length"] = tickets["description"].str.len()
    tickets["word_count"]  = tickets["description"].str.split().str.len()

    # Category and priority as integers
    le = LabelEncoder()
    tickets["category_enc"] = le.fit_transform(tickets["category"].astype(str))
    cat_map = dict(zip(le.classes_, le.transform(le.classes_)))
    log.info(f"  Category encoding: {cat_map}")

    prio_map = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    tickets["priority_enc"] = tickets["priority"].map(prio_map).fillna(1).astype(int)

    # Merge customer features
    cust_feats = customers[["customer_id","loyalty_tier","segment","age"]]
    feat = tickets.merge(cust_feats, on="customer_id", how="left")

    save(feat, "tickets")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 55)
    log.info("  Nexalyze — Feature Engineering Pipeline")
    log.info("=" * 55)

    build_customer_features()
    build_product_features()
    build_transaction_features()
    build_inventory_features()
    build_intent_features()
    build_ticket_features()

    log.info("=" * 55)
    log.info("  Feature engineering complete!")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
