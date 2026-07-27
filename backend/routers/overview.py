"""
Executive overview — the console's landing dashboard.

Everything here is aggregated in SQLite rather than in pandas. These are
whole-estate rollups over the transaction log, and pushing them down to the
database keeps the landing page fast enough to render on every visit instead of
pulling hundreds of thousands of rows into the API process to group them there.

Nothing on this surface is a constant. Every figure, threshold and axis is
derived from the data, including the quadrant medians the category view splits
on — a fixed "high growth" line would be meaningless against a dataset whose
scale changes with the build.
"""
from __future__ import annotations

from fastapi import APIRouter, Query

from backend.utils import db

router = APIRouter(prefix="/api/v1/overview", tags=["Executive Overview"])

# Revenue net of what the goods cost. transactions carries what was charged;
# cost sits on the product, so margin needs the join in every rollup below.
_MARGIN = "SUM(t.total_amount - (t.quantity * p.cost_price))"

# Every rollup reads the same three dimensions, so they share one FROM clause.
# Region lives on the customer, not the transaction, which is why the customer
# join is always present even when no region filter is applied.
_FROM = """
    FROM transactions t
    JOIN products  p ON p.product_id  = t.product_id
    JOIN customers c ON c.customer_id = t.customer_id
"""


def _filters(category: str | None, channel: str | None, region: str | None,
             year: str | None = None):
    """Build the shared WHERE fragment. 'All' and empty both mean unfiltered."""
    clauses, params = [], []
    for column, value in (("p.category", category), ("t.channel", channel),
                          ("c.region", region)):
        if value and value.lower() != "all":
            clauses.append(f"{column} = ?")
            params.append(value)
    if year and year.lower() != "all":
        clauses.append("strftime('%Y', t.transaction_date) = ?")
        params.append(year)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


@router.get("/filters")
def filter_options():
    """Every option comes from the data, so a filter can never offer an empty slice."""
    def col(sql):
        return db.query(sql, parse_dates=False).iloc[:, 0].dropna().tolist()

    return {
        "categories": col("SELECT DISTINCT category FROM products "
                          "WHERE category IS NOT NULL ORDER BY category"),
        "channels": col("SELECT DISTINCT channel FROM transactions "
                        "WHERE channel IS NOT NULL ORDER BY channel"),
        "regions": col("SELECT DISTINCT region FROM customers "
                       "WHERE region IS NOT NULL ORDER BY region"),
        "years": [str(y) for y in col(
            "SELECT DISTINCT strftime('%Y', transaction_date) FROM transactions "
            "WHERE transaction_date IS NOT NULL ORDER BY 1")],
    }


@router.get("/estate")
def estate(category: str = Query(None), channel: str = Query(None),
           region: str = Query(None), year: str = Query(None)):
    """Headline position for the current filter slice."""
    where, params = _filters(category, channel, region, year)

    head = db.query(f"""
        SELECT
            COUNT(*)                              AS transactions,
            COALESCE(SUM(t.total_amount), 0)      AS revenue,
            COALESCE(SUM(t.quantity), 0)          AS units,
            COALESCE({_MARGIN}, 0)                AS margin,
            COUNT(DISTINCT t.customer_id)         AS buyers,
            COUNT(DISTINCT t.product_id)          AS skus_sold,
            MIN(t.transaction_date)               AS first_txn,
            MAX(t.transaction_date)               AS last_txn,
            COALESCE(AVG(t.total_amount), 0)      AS avg_basket,
            COALESCE(AVG(t.discount_pct), 0)      AS avg_discount,
            COALESCE(SUM(CASE WHEN t.is_returned = 1
                              THEN t.total_amount ELSE 0 END), 0) AS returned
        {_FROM}
        WHERE 1=1 {where}
    """, params, parse_dates=False).iloc[0]

    revenue = float(head["revenue"])
    margin = float(head["margin"])

    return {
        "revenue": round(revenue, 2),
        "margin": round(margin, 2),
        "margin_pct": round(margin / revenue * 100, 1) if revenue else 0.0,
        "units": int(head["units"]),
        "transactions": int(head["transactions"]),
        "avg_basket": round(float(head["avg_basket"]), 2),
        "avg_discount_pct": round(float(head["avg_discount"]) * 100, 1),
        "buyers": int(head["buyers"]),
        "skus_sold": int(head["skus_sold"]),
        "return_rate_pct": round(float(head["returned"]) / revenue * 100, 1) if revenue else 0.0,
        "period_start": str(head["first_txn"])[:10] if head["first_txn"] else "—",
        "period_end": str(head["last_txn"])[:10] if head["last_txn"] else "—",
        "customers": db.row_count("customers"),
        "products": db.row_count("products"),
        "stores": db.row_count("stores"),
        "warehouses": db.row_count("warehouses"),
        "suppliers": db.row_count("suppliers"),
    }


@router.get("/revenue-trend")
def revenue_trend(months: int = Query(36, ge=6, le=120),
                  category: str = Query(None), channel: str = Query(None),
                  region: str = Query(None), year: str = Query(None)):
    """Monthly revenue, margin and units for the current filter slice."""
    where, params = _filters(category, channel, region, year)

    rows = db.query(f"""
        SELECT strftime('%Y-%m', t.transaction_date) AS month,
               SUM(t.total_amount)          AS revenue,
               {_MARGIN}                    AS margin,
               SUM(t.quantity)              AS units,
               COUNT(*)                     AS transactions,
               COUNT(DISTINCT t.customer_id) AS buyers
        {_FROM}
        WHERE t.transaction_date IS NOT NULL {where}
        GROUP BY month
        ORDER BY month
    """, params, parse_dates=False).tail(months)

    # The channel split ignores the channel filter — with one channel selected a
    # breakdown by channel is a single 100% bar. Every other filter still applies,
    # so it stays a genuine split of the slice being looked at.
    ch_where, ch_params = _filters(category, None, region, year)
    channel = db.query(f"""
        SELECT t.channel AS channel, SUM(t.total_amount) AS revenue, COUNT(*) AS transactions
        {_FROM}
        WHERE t.channel IS NOT NULL {ch_where}
        GROUP BY t.channel ORDER BY revenue DESC
    """, ch_params, parse_dates=False)

    series = [{
        "month": r["month"],
        "revenue": round(float(r["revenue"] or 0), 2),
        "margin": round(float(r["margin"] or 0), 2),
        "units": int(r["units"] or 0),
        "transactions": int(r["transactions"] or 0),
        "buyers": int(r["buyers"] or 0),
    } for r in rows.to_dict(orient="records")]

    # Best and worst month are read off the series rather than assumed, so the
    # callout stays correct whatever window the caller asked for.
    peak = max(series, key=lambda s: s["revenue"], default=None)
    trough = min(series, key=lambda s: s["revenue"], default=None)

    return {
        "series": series,
        "channels": [{"channel": c["channel"],
                      "revenue": round(float(c["revenue"] or 0), 2),
                      "transactions": int(c["transactions"] or 0)}
                     for c in channel.to_dict(orient="records")],
        "peak_month": peak["month"] if peak else None,
        "peak_revenue": peak["revenue"] if peak else 0,
        "trough_month": trough["month"] if trough else None,
    }


@router.get("/category-performance")
def category_performance(category: str = Query(None), channel: str = Query(None),
                         region: str = Query(None)):
    """
    Revenue, growth and margin per category — the inputs to a growth/share view.

    Picking a category drills into its sub-categories rather than returning a
    single bubble, so the filter deepens the view instead of emptying it.

    There is deliberately no year parameter: growth here *is* a year-on-year
    comparison, and restricting the rows to one year would leave nothing to
    compare against. Growth anchors to the last two years present in the data,
    not to the wall clock — this dataset ends in 2024, so a "last 12 months from
    today" window would return nothing at all.
    """
    years = db.query(
        "SELECT DISTINCT strftime('%Y', transaction_date) AS y FROM transactions "
        "WHERE transaction_date IS NOT NULL ORDER BY y", parse_dates=False)["y"].tolist()
    if not years:
        return {"categories": [], "latest_year": None, "prior_year": None}

    latest = years[-1]
    prior = years[-2] if len(years) > 1 else None

    drilled = bool(category and category.lower() != "all")
    dimension = "p.sub_category" if drilled else "p.category"

    where, params = _filters(category, channel, region)
    rows = db.query(f"""
        SELECT {dimension} AS label,
               SUM(t.total_amount)  AS revenue,
               SUM(t.quantity)      AS units,
               {_MARGIN}            AS margin,
               COUNT(DISTINCT t.customer_id) AS buyers,
               COUNT(DISTINCT t.product_id)  AS skus,
               SUM(CASE WHEN strftime('%Y', t.transaction_date) = ?
                        THEN t.total_amount ELSE 0 END) AS rev_latest,
               SUM(CASE WHEN strftime('%Y', t.transaction_date) = ?
                        THEN t.total_amount ELSE 0 END) AS rev_prior
        {_FROM}
        WHERE {dimension} IS NOT NULL {where}
        GROUP BY {dimension}
        ORDER BY revenue DESC
    """, [latest, prior or "—", *params], parse_dates=False)

    total_rev = float(rows["revenue"].sum()) or 1.0
    cats = []
    for r in rows.to_dict(orient="records"):
        rev = float(r["revenue"] or 0)
        last, before = float(r["rev_latest"] or 0), float(r["rev_prior"] or 0)
        cats.append({
            "category": r["label"],
            "revenue": round(rev, 2),
            "revenue_share_pct": round(rev / total_rev * 100, 1),
            "units": int(r["units"] or 0),
            "margin": round(float(r["margin"] or 0), 2),
            "margin_pct": round(float(r["margin"] or 0) / rev * 100, 1) if rev else 0.0,
            "buyers": int(r["buyers"] or 0),
            "skus": int(r["skus"] or 0),
            "growth_pct": round((last - before) / before * 100, 1) if before else 0.0,
        })

    # The split lines are medians of what is actually there, so the four
    # quadrants always carry categories instead of collapsing into one corner.
    def _median(key):
        vals = sorted(c[key] for c in cats)
        if not vals:
            return 0.0
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2

    return {
        "categories": cats,
        "latest_year": latest,
        "prior_year": prior,
        "median_revenue": round(_median("revenue"), 2),
        "median_growth_pct": round(_median("growth_pct"), 1),
        "total_revenue": round(total_rev, 2),
        "dimension": "sub-category" if drilled else "category",
        "drilled_into": category if drilled else None,
    }


@router.get("/seasonality")
def seasonality(category: str = Query(None), channel: str = Query(None),
                region: str = Query(None), year: str = Query(None)):
    """
    Revenue intensity by calendar month and weekday.

    Averaged per occurrence, not summed: months and weekdays do not appear an
    equal number of times across the window, and a raw sum would read as a
    trading-day artefact rather than a seasonal pattern.
    """
    where, params = _filters(category, channel, region, year)
    rows = db.query(f"""
        SELECT CAST(strftime('%m', t.transaction_date) AS INTEGER) AS month,
               CAST(strftime('%w', t.transaction_date) AS INTEGER) AS weekday,
               SUM(t.total_amount) AS revenue,
               COUNT(DISTINCT date(t.transaction_date)) AS days
        {_FROM}
        WHERE t.transaction_date IS NOT NULL {where}
        GROUP BY month, weekday
    """, params, parse_dates=False)

    cells = [{
        "month": int(r["month"]),
        "weekday": int(r["weekday"]),          # 0 = Sunday, as SQLite reports it
        "revenue": round(float(r["revenue"] or 0), 2),
        "days": int(r["days"] or 0),
        "revenue_per_day": round(float(r["revenue"] or 0) / max(1, int(r["days"] or 1)), 2),
    } for r in rows.to_dict(orient="records")]

    best = max(cells, key=lambda c: c["revenue_per_day"], default=None)
    return {
        "cells": cells,
        "peak_month": best["month"] if best else None,
        "peak_weekday": best["weekday"] if best else None,
        "peak_revenue_per_day": best["revenue_per_day"] if best else 0,
    }


@router.get("/top-movers")
def top_movers(limit: int = Query(6, ge=1, le=25),
               category: str = Query(None), channel: str = Query(None),
               region: str = Query(None), year: str = Query(None)):
    """
    Products carrying the most revenue in the current slice.

    With no year selected this ranks over the whole history rather than
    defaulting to the latest year — every other panel on the dashboard treats an
    unset filter as "everything", and quietly narrowing to one year here made
    the table disagree with the totals above it.
    """
    scoped = bool(year and year.lower() != "all")
    where, params = _filters(category, channel, region, year)
    rows = db.query(f"""
        SELECT p.product_name, p.category, p.brand,
               SUM(t.total_amount) AS revenue,
               SUM(t.quantity)     AS units,
               {_MARGIN}           AS margin
        {_FROM}
        WHERE 1=1 {where}
        GROUP BY t.product_id
        ORDER BY revenue DESC
        LIMIT ?
    """, [*params, limit], parse_dates=False)

    return {
        "year": year if scoped else "all years",
        "products": [{
            "product_name": r["product_name"],
            "category": r["category"],
            "brand": r["brand"],
            "revenue": round(float(r["revenue"] or 0), 2),
            "units": int(r["units"] or 0),
            "margin_pct": round(float(r["margin"] or 0) / float(r["revenue"]) * 100, 1)
                          if r["revenue"] else 0.0,
        } for r in rows.to_dict(orient="records")],
    }
