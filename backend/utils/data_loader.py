"""
CalRetail — Data loader utility

Thin, table-named accessors over the SQLite database (see ``backend.utils.db``).
The ``get_*`` functions return whole memoised tables and keep the exact contract
the CSV loader had, so every service that predates the database still works.

New code that only needs part of a table should prefer the pushdown helpers at
the bottom of this module, or ``db.query`` directly — loading 300k transactions
to answer a question about one customer is what the indexes exist to avoid.
"""
from backend.utils import db
from backend.utils.db import load_df  # re-exported: services import it from here


def get_customers():
    return load_df("customers")

def get_products():
    return load_df("products")

def get_transactions():
    return load_df("transactions")

def get_inventory():
    return load_df("inventory")

def get_reviews():
    return load_df("customer_reviews")

def get_tickets():
    return load_df("support_tickets")

def get_browsing():
    return load_df("browsing_history")

def get_pricing_history():
    return load_df("pricing_history")

def get_competitor_pricing():
    return load_df("competitor_pricing")

def get_promotions():
    return load_df("promotions")

def get_campaigns():
    return load_df("marketing_campaigns")

def get_orders():
    return load_df("orders")

def get_stores():
    return load_df("stores")

def get_warehouses():
    return load_df("warehouses")

def get_suppliers():
    return load_df("suppliers")

def get_shipments():
    return load_df("shipments")

def get_returns():
    return load_df("returns")

def get_inventory_movements():
    return load_df("inventory_movements")

def get_wishlist():
    return load_df("wishlist")

def get_feature_customers():
    return load_df("feature_customers")

def get_feature_products():
    return load_df("feature_products")

def get_feature_daily_sales():
    return load_df("feature_daily_sales")

def get_feature_inventory_health():
    return load_df("feature_inventory_health")

def get_feature_buying_intent():
    return load_df("feature_buying_intent")

def get_feature_tickets():
    return load_df("feature_tickets")


# ── Indexed lookups ──────────────────────────────────────────────────────────
# These hit the indexes built by notebooks.build_db rather than scanning a
# memoised frame, which is the difference between a few milliseconds and
# materialising a whole event log per request.

def customer(customer_id: str):
    """One customer row as a dict, or None."""
    df = db.read_table("customers", where="customer_id = ?", params=(customer_id,), limit=1)
    return None if df.empty else df.iloc[0].to_dict()


def product(product_id: str):
    df = db.read_table("products", where="product_id = ?", params=(product_id,), limit=1)
    return None if df.empty else df.iloc[0].to_dict()


def customer_transactions(customer_id: str, limit: int | None = None):
    return db.read_table("transactions", where="customer_id = ?", params=(customer_id,),
                         order_by="transaction_date DESC", limit=limit)


def product_transactions(product_id: str, limit: int | None = None):
    return db.read_table("transactions", where="product_id = ?", params=(product_id,),
                         order_by="transaction_date DESC", limit=limit)


def customer_orders(customer_id: str, limit: int | None = None):
    return db.read_table("orders", where="customer_id = ?", params=(customer_id,),
                         order_by="order_date DESC", limit=limit)


def product_inventory(product_id: str):
    return db.read_table("inventory", where="product_id = ?", params=(product_id,))


def categories():
    """Distinct product categories, straight from SQLite."""
    return db.query(
        "SELECT DISTINCT category FROM products "
        "WHERE category IS NOT NULL ORDER BY category"
    )["category"].tolist()
