"""
Nexalyze — Data Cleaning Script
Reads the raw build database, applies cleaning rules, and writes the cleaned
tables into the shipped database (data/calretail.db).
Run via: python -m notebooks.clean_data  OR  python -m notebooks.build_db
"""

import warnings
import logging

import numpy as np
import pandas as pd

from notebooks.pipeline_io import load_raw, save_processed

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("nexalyze.cleaning")


def load(name: str) -> pd.DataFrame:
    df = load_raw(name)
    log.info(f"Loaded {name}  — {len(df):,} rows, {df.shape[1]} cols")
    return df


def save(df: pd.DataFrame, name: str) -> None:
    save_processed(df, name)
    log.info(f"Saved  {name}  — {len(df):,} rows")


def report_nulls(df: pd.DataFrame, name: str) -> None:
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        log.info(f"  [{name}] Nulls:\n{nulls}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-table cleaning functions
# ─────────────────────────────────────────────────────────────────────────────

def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "customers")
    df = df.drop_duplicates(subset=["customer_id"])
    df["age"] = df["age"].clip(lower=16, upper=90)
    df["shopping_frequency"] = df["shopping_frequency"].clip(lower=0).fillna(2.0)
    df["gender"] = df["gender"].fillna("Unknown")
    df["income_bracket"] = df["income_bracket"].fillna("<2L")
    df["registration_date"] = pd.to_datetime(df["registration_date"], errors="coerce")
    df["registration_date"] = df["registration_date"].fillna(pd.Timestamp("2020-01-01"))
    df["email_opt_in"] = df["email_opt_in"].fillna(False).astype(bool)
    df["sms_opt_in"]   = df["sms_opt_in"].fillna(False).astype(bool)
    df["app_installed"] = df["app_installed"].fillna(False).astype(bool)
    df["is_active"] = df["is_active"].fillna(True).astype(bool)
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "products")
    df = df.drop_duplicates(subset=["product_id"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(499)
    df["mrp"]   = pd.to_numeric(df["mrp"],   errors="coerce").fillna(df["price"] * 1.3)
    df["cost_price"] = pd.to_numeric(df["cost_price"], errors="coerce").fillna(df["price"] * 0.45)
    df["price"] = df["price"].clip(lower=1)
    df["mrp"]   = np.maximum(df["mrp"], df["price"])
    df["cost_price"] = np.minimum(df["cost_price"], df["price"])
    df["margin_pct"] = ((df["price"] - df["cost_price"]) / df["price"] * 100).round(2)
    df["launch_date"] = pd.to_datetime(df["launch_date"], errors="coerce")
    df["is_active"] = df["is_active"].fillna(True).astype(bool)
    df["style_tag"] = df["style_tag"].fillna("casual")
    df["season"]    = df["season"].fillna("All Season")
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "transactions")
    df = df.drop_duplicates(subset=["transaction_id"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["transaction_date", "customer_id", "product_id"])
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).clip(lower=1)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").clip(lower=0)
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").clip(0, 0.90).fillna(0)
    df["final_price"]  = pd.to_numeric(df["final_price"], errors="coerce").clip(lower=0)
    df["total_amount"] = (df["final_price"] * df["quantity"]).round(2)
    df["channel"] = df["channel"].fillna("Online")
    df["payment_method"] = df["payment_method"].fillna("UPI")
    df["is_returned"] = df["is_returned"].fillna(False).astype(bool)
    # Derived time features
    df["year"]        = df["transaction_date"].dt.year
    df["month"]       = df["transaction_date"].dt.month
    df["day_of_week"] = df["transaction_date"].dt.day_name()
    df["week_number"] = df["transaction_date"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["transaction_date"].dt.quarter
    return df


def clean_browsing_history(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "browsing_history")
    df = df.drop_duplicates(subset=["browse_id"])
    df["dwell_seconds"] = pd.to_numeric(df["dwell_seconds"], errors="coerce").clip(0, 3600).fillna(30)
    df["page_views"]    = pd.to_numeric(df["page_views"],    errors="coerce").clip(1, 100).fillna(1)
    df["timestamp"]     = pd.to_datetime(df["timestamp"], errors="coerce")
    df["added_to_cart"] = df["added_to_cart"].fillna(False).astype(bool)
    df["purchased"]     = df["purchased"].fillna(False).astype(bool)
    df = df.dropna(subset=["customer_id", "product_id"])
    return df


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "orders")
    df = df.drop_duplicates(subset=["order_id"])
    df["order_date"]           = pd.to_datetime(df["order_date"], errors="coerce")
    df["estimated_delivery"]   = pd.to_datetime(df["estimated_delivery"], errors="coerce")
    df["actual_delivery"]      = pd.to_datetime(df["actual_delivery"],    errors="coerce")
    df = df.dropna(subset=["order_date", "customer_id", "product_id"])
    df["quantity"]     = pd.to_numeric(df["quantity"],     errors="coerce").fillna(1).clip(1)
    df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce").clip(0)
    df["shipping_fee"] = pd.to_numeric(df["shipping_fee"], errors="coerce").fillna(0)
    df["status"]       = df["status"].fillna("Placed")
    # Delivery delay in days
    df["delivery_delay_days"] = (df["actual_delivery"] - df["estimated_delivery"]).dt.days.fillna(0)
    return df


def clean_inventory(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "inventory")
    df = df.drop_duplicates(subset=["inventory_id"])
    df["stock_qty"]    = pd.to_numeric(df["stock_qty"],    errors="coerce").clip(0).fillna(0)
    df["reorder_point"]= pd.to_numeric(df["reorder_point"],errors="coerce").clip(0).fillna(20)
    df["max_stock"]    = pd.to_numeric(df["max_stock"],    errors="coerce").clip(0).fillna(200)
    df["days_on_hand"] = pd.to_numeric(df["days_on_hand"], errors="coerce").clip(0).fillna(0)
    df["stockout_risk"] = (df["stock_qty"] <= df["reorder_point"]).astype(int)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


def clean_support_tickets(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "support_tickets")
    df = df.drop_duplicates(subset=["ticket_id"])
    df["created_at"]  = pd.to_datetime(df["created_at"],  errors="coerce")
    df["resolved_at"] = pd.to_datetime(df["resolved_at"], errors="coerce")
    df["resolution_hrs"] = (
        (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600
    ).clip(0).fillna(0).round(1)
    df["csat_score"] = pd.to_numeric(df["csat_score"], errors="coerce")
    df["status"]     = df["status"].fillna("Open")
    df["priority"]   = df["priority"].fillna("Medium")
    df["category"]   = df["category"].fillna("Order Issue")
    df["description"]= df["description"].fillna("").str.strip()
    return df


def clean_pricing_history(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "pricing_history")
    df = df.drop_duplicates(subset=["price_id"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").clip(1)
    df["effective_date"] = pd.to_datetime(df["effective_date"], errors="coerce")
    df = df.dropna(subset=["product_id", "effective_date"])
    return df


def clean_competitor_pricing(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "competitor_pricing")
    df = df.drop_duplicates(subset=["comp_price_id"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce").clip(1)
    df["scraped_date"] = pd.to_datetime(df["scraped_date"], errors="coerce")
    df["in_stock"] = df["in_stock"].fillna(True).astype(bool)
    return df


def clean_customer_reviews(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "customer_reviews")
    df = df.drop_duplicates(subset=["review_id"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").clip(1, 5).fillna(3)
    df["review_text"] = df["review_text"].fillna("").str.strip()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df["sentiment_label"] = df["sentiment_label"].fillna("Neutral")
    df["helpful_votes"]   = pd.to_numeric(df["helpful_votes"], errors="coerce").clip(0).fillna(0)
    df["is_verified"]     = df["is_verified"].fillna(False).astype(bool)
    df = df[df["review_text"].str.len() > 0]  # Drop empty reviews
    return df


def clean_returns(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "returns")
    df = df.drop_duplicates(subset=["return_id"])
    df["return_date"]   = pd.to_datetime(df["return_date"],   errors="coerce")
    df["refund_amount"] = pd.to_numeric(df["refund_amount"],  errors="coerce").clip(0).fillna(0)
    df["reason"]  = df["reason"].fillna("Other")
    df["status"]  = df["status"].fillna("Requested")
    return df


def clean_promotions(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "promotions")
    df = df.drop_duplicates(subset=["promo_id"])
    df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").clip(0, 0.90).fillna(0.10)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    df["budget"] = pd.to_numeric(df["budget"], errors="coerce").clip(0).fillna(10_000)
    df["is_active"] = df["is_active"].fillna(False).astype(bool)
    # Duration in days
    df["duration_days"] = (df["end_date"] - df["start_date"]).dt.days.clip(0).fillna(7)
    return df


def clean_marketing_campaigns(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "marketing_campaigns")
    df = df.drop_duplicates(subset=["campaign_id"])
    for col in ["budget","amount_spent","impressions","clicks","conversions"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0).fillna(0)
    df["ctr"]  = np.where(df["impressions"] > 0,
                          (df["clicks"] / df["impressions"] * 100).round(4), 0)
    df["cvr"]  = np.where(df["clicks"] > 0,
                          (df["conversions"] / df["clicks"] * 100).round(4), 0)
    df["roas"] = np.where(df["amount_spent"] > 0,
                          (df["conversions"] * 500 / df["amount_spent"]).round(2), 0)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]   = pd.to_datetime(df["end_date"],   errors="coerce")
    return df


def clean_shipments(df: pd.DataFrame) -> pd.DataFrame:
    report_nulls(df, "shipments")
    df = df.drop_duplicates(subset=["shipment_id"])
    df["ship_date"]       = pd.to_datetime(df["ship_date"],       errors="coerce")
    df["expected_arrival"]= pd.to_datetime(df["expected_arrival"], errors="coerce")
    df["actual_arrival"]  = pd.to_datetime(df["actual_arrival"],  errors="coerce")
    df["freight_cost"]    = pd.to_numeric(df["freight_cost"],     errors="coerce").clip(0).fillna(0)
    df["quantity"]        = pd.to_numeric(df["quantity"],         errors="coerce").clip(1).fillna(1)
    df["delay_days"] = (df["actual_arrival"] - df["expected_arrival"]).dt.days.fillna(0)
    return df


def clean_pass_through(name: str) -> pd.DataFrame:
    """For tables needing only de-duplication and date parsing."""
    df = load(name)
    df = df.drop_duplicates()
    # Parse any _date columns
    for col in df.columns:
        if ("date" in col.lower() or "time" in col.lower()) and "days" not in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

CLEANERS = {
    "customers":           clean_customers,
    "products":            clean_products,
    "transactions":        clean_transactions,
    "browsing_history":    clean_browsing_history,
    "orders":              clean_orders,
    "inventory":           clean_inventory,
    "support_tickets":     clean_support_tickets,
    "pricing_history":     clean_pricing_history,
    "competitor_pricing":  clean_competitor_pricing,
    "customer_reviews":    clean_customer_reviews,
    "returns":             clean_returns,
    "promotions":          clean_promotions,
    "marketing_campaigns": clean_marketing_campaigns,
    "shipments":           clean_shipments,
}

PASS_THROUGH = [
    "stores","warehouses","suppliers","employees","holiday_calendar","weather",
    "customer_sessions","wishlist","shopping_cart","search_history",
    "inventory_movements",
]


def main():
    log.info("=" * 55)
    log.info("  Nexalyze — Data Cleaning Pipeline")
    log.info("=" * 55)

    for name, cleaner in CLEANERS.items():
        df_raw = load(name)
        df_clean = cleaner(df_raw)
        save(df_clean, name)

    for name in PASS_THROUGH:
        df = clean_pass_through(name)
        save(df, name)

    log.info("=" * 55)
    log.info("  Cleaning complete — processed data saved to data/processed/")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
