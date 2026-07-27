"""
Module 1 — Customer Experience service layer.

Thin wrappers over backend.capabilities.*, memoised because the underlying
database is read-only and the console re-requests the same slices.
"""

from backend.utils.cache import ttl_cache


@ttl_cache()
def get_recommendations(customer_id: str, top_n: int = 10) -> list[dict]:
    from backend.capabilities import personalised_recommendations as mod
    res = mod.get_recommendations(customer_id, top_n=top_n)
    recs = res.get("recommendations", [])

    from backend.utils.data_loader import get_products
    prods = get_products()
    brand_map = dict(zip(prods["product_id"], prods["brand"]))
    for r in recs:
        r["brand"] = brand_map.get(r.get("product_id"), "Unknown")
    return recs


@ttl_cache()
def get_recommendations_debug(customer_id: str, top_n: int = 10) -> dict:
    """
    Admin view: why each product was recommended, based on reviews,
    past purchases, and what similar customers actually bought.
    No raw internal scores exposed — all reason-based for human review.
    """
    import pandas as pd
    from backend.utils.data_loader import get_products, get_reviews, get_customers

    from backend.capabilities import personalised_recommendations as mod
    cust_df   = mod.cust
    tx_df     = mod.tx
    prod_df   = mod.prod
    # The notebook's implicit-feedback matrix is a sparse CSR with its labels
    # held alongside it, so this reads positions rather than pandas labels.
    matrix     = mod.matrix
    cust_index = mod.cust_index
    prod_index = mod.prod_index
    category_boost = mod.category_boost

    # ── Product enrichment maps ───────────────────────────────────────────────
    prods_full = get_products()
    brand_map  = dict(zip(prods_full["product_id"], prods_full["brand"]))

    reviews = get_reviews()
    rating_stats = reviews.groupby("product_id")["rating"].agg(["mean", "count"])

    bestsellers = tx_df.groupby("product_id")["quantity"].sum()

    # ── Customer profile — resolve from raw customers CSV ────────────────────
    profile: dict = {}
    cust_src = get_customers()  # always load from raw CSV for full column coverage
    cust_row = cust_src[cust_src["customer_id"] == customer_id]
    if not cust_row.empty:
        row = cust_row.iloc[0]
        segment   = str(row.get("segment", "") or "").strip() or "Unknown"
        loyalty   = str(row.get("loyalty_tier", "") or "").strip() or "Unknown"
        pref_cat  = str(row.get("preferred_category", "") or "").strip() or None
        profile = {
            "name":               str(row.get("name", customer_id)),
            "segment":            segment,
            "loyalty_tier":       loyalty,
            "preferred_category": pref_cat or "Unknown",
            "region":             str(row.get("region", "Unknown") or "Unknown"),
            "city":               str(row.get("city", "Unknown") or "Unknown"),
        }
    else:
        pref_cat = None
        profile  = {
            "name": customer_id, "segment": "Unknown", "loyalty_tier": "Unknown",
            "preferred_category": "Unknown", "region": "Unknown", "city": "Unknown",
        }

    # ── Purchase history ──────────────────────────────────────────────────────
    hist = tx_df[tx_df["customer_id"] == customer_id].merge(
        prod_df[["product_id", "product_name", "category"]], on="product_id", how="left"
    )
    purchase_history   = []
    category_affinity  = []
    already_purchased_pids = set()

    if not hist.empty:
        hist = hist.sort_values("transaction_date", ascending=False)
        hist["brand"] = hist["product_id"].map(brand_map)
        already_purchased_pids = set(hist["product_id"].unique())
        for _, r in hist.iterrows():
            purchase_history.append({
                "transaction_id":   r["transaction_id"],
                "product_id":       r["product_id"],
                "product_name":     r["product_name"],
                "category":         r["category"],
                "brand":            r["brand"],
                "quantity":         int(r["quantity"]),
                "unit_price":       float(r["unit_price"]),
                "final_price":      float(r["final_price"]),
                "total_amount":     float(r["total_amount"]),
                "transaction_date": str(r["transaction_date"])[:10],
            })
        cat_units = hist.groupby("category")["quantity"].sum().sort_values(ascending=False)
        total_units = float(cat_units.sum()) or 1.0
        category_affinity = [
            {"category": cat,
             "purchase_pct": round(float(units) / total_units * 100, 1),
             "units_purchased": int(units)}
            for cat, units in cat_units.items()
        ]

    # ── Helper: reason tags for a product ────────────────────────────────────
    def _reasons(pid: str, category: str, avg_rating: float,
                 review_count: int, is_personalized: bool) -> list[str]:
        reasons = []
        if is_personalized:
            reasons.append("Bought by similar customers")
        if pid in already_purchased_pids:
            reasons.append("In your purchase history")
        elif category == pref_cat:
            reasons.append(f"Matches your preferred category ({pref_cat})")
        if avg_rating >= 4.0 and review_count >= 5:
            reasons.append(f"Highly rated ({avg_rating:.1f}★, {review_count} reviews)")
        if not reasons:
            reasons.append("Popular among shoppers")
        return reasons

    # ── Fallback rows ─────────────────────────────────────────────────────────
    def _fallback_rows(algo_label: str) -> list[dict]:
        pool = prod_df[prod_df["product_id"].isin(bestsellers.index)].copy()
        pool["rank_score"] = pool["product_id"].map(bestsellers)
        if pref_cat:
            pool = pd.concat([
                pool[pool["category"] == pref_cat],
                pool[pool["category"] != pref_cat]
            ])
        pool = pool.drop_duplicates("product_id").head(top_n * 3).nlargest(top_n, "rank_score")
        rows = []
        for _, r in pool.iterrows():
            pid       = r["product_id"]
            avg_r     = round(float(rating_stats["mean"].get(pid, 0.0)), 1)
            rev_count = int(rating_stats["count"].get(pid, 0))
            rows.append({
                "product_id":          pid,
                "product_name":        r["product_name"],
                "category":            r["category"],
                "brand":               brand_map.get(pid, "Unknown"),
                "price":               float(r["price"]),
                "avg_rating":          avg_r,
                "review_count":        rev_count,
                "freq_by_similar":     0,
                "reasons":             _reasons(pid, r["category"], avg_r, rev_count, False),
                "is_personalized":     False,
                "algorithm":           algo_label,
            })
        return rows

    # ── Main CF path ──────────────────────────────────────────────────────────
    similar_customers: list[dict] = []

    if customer_id not in cust_index:
        algorithm = "Bestseller Fallback — Cold Start (no purchase history)"
        recs = _fallback_rows(algorithm)
    else:
        from sklearn.metrics.pairwise import cosine_similarity

        # One row against the rest, rather than a stored 10k x 10k matrix.
        row_pos = cust_index.get_loc(customer_id)
        sims = pd.Series(
            cosine_similarity(matrix[row_pos], matrix)[0], index=cust_index
        ).drop(customer_id).nlargest(20)
        sims = sims[sims > 0]

        if len(sims) == 0:
            algorithm = "Bestseller Fallback — No Similar Shoppers Found"
            recs = _fallback_rows(algorithm)
        else:
            name_map = dict(zip(cust_df["customer_id"], cust_df["name"]))

            # Similar customers list for the chart
            similar_customers = [
                {
                    "customer_id": cid,
                    "name":        name_map.get(cid, cid),
                    "similarity":  round(float(s), 4),
                }
                for cid, s in sims.head(10).items()
            ]

            # Non-zero columns of the customer's own row are what they have.
            already_owned = prod_index[matrix[row_pos].indices]

            # Weighted collaborative filtering scores: (20 x P)^T . (20,)
            weighted = pd.Series(
                matrix[cust_index.get_indexer(sims.index)]
                .T.dot(sims.to_numpy(dtype="float32")) / sims.sum(),
                index=prod_index,
            )
            candidates = weighted.drop(index=already_owned, errors="ignore")

            # Frequency — how many unique similar customers bought each product
            sim_ids = sims.index.tolist()
            sim_tx  = tx_df[tx_df["customer_id"].isin(sim_ids)]
            freq_map = sim_tx.groupby("product_id")["customer_id"].nunique().to_dict()

            algorithm = "Collaborative Filtering + Category Boost"
            recs = []
            for pid, raw_score in candidates.nlargest(top_n * 3).items():
                if raw_score <= 0:
                    continue
                p_rows = prod_df[prod_df["product_id"] == pid]
                if p_rows.empty:
                    continue
                p_info    = p_rows.iloc[0]
                avg_r     = round(float(rating_stats["mean"].get(pid, 0.0)), 1)
                rev_count = int(rating_stats["count"].get(pid, 0))
                freq      = int(freq_map.get(pid, 0))
                recs.append({
                    "product_id":      pid,
                    "product_name":    p_info["product_name"],
                    "category":        p_info["category"],
                    "brand":           brand_map.get(pid, "Unknown"),
                    "price":           float(p_info["price"]),
                    "avg_rating":      avg_r,
                    "review_count":    rev_count,
                    "freq_by_similar": freq,
                    "reasons":         _reasons(pid, p_info["category"], avg_r, rev_count, True),
                    "is_personalized": True,
                    "algorithm":       algorithm,
                })

            recs = sorted(recs, key=lambda x: (x["freq_by_similar"], x["avg_rating"]), reverse=True)[:top_n]
            if not recs:
                algorithm = "Bestseller Fallback — No Positive CF Candidates"
                recs = _fallback_rows(algorithm)
                similar_customers = []

    return {
        "customer_id":     customer_id,
        "customer_name":   (profile or {}).get("name") or customer_id,
        "profile":         profile,
        "purchase_history":   purchase_history,
        "category_affinity":  category_affinity,
        "recommendations":    recs,
        "similar_customers":  similar_customers,
        "algorithm":          algorithm,
    }


def buying_assistant_query(customer_id: str, message: str) -> dict:
    from backend.capabilities import conversational_buying_assistant as mod
    res = mod.process_chat_message(customer_id, message)
    return {
        "intent":             res.get("intent", "browse"),
        "detected_category":  res.get("category"),
        "max_price":          res.get("price_limit"),
        "response":           res.get("response", ""),
        "product_suggestions":res.get("suggestions", []),
    }


@ttl_cache()
def get_next_best_offer(customer_id: str) -> dict:
    from backend.capabilities import next_best_offer as mod
    res = mod.resolve_nbo(customer_id)

    opt = res.get("recommended_offer", {})
    disc_str = opt.get("discount", "0%").replace("%", "")
    try:
        disc_pct = float(disc_str)
    except ValueError:
        disc_pct = 0.0

    p_type = opt.get("promo_type", "Discount (Product Boost)")
    return {
        "promo_type":      p_type,
        "discount_pct":    disc_pct,
        "predicted_uplift": float(res.get("uplift_pct", 0.0)),
        "confidence":      float(res.get("confidence_score", 0.0)),
        "message":         (
            f"Send push notification recommending product "
            f"{opt.get('product_id')} with {opt.get('discount')} discount."
        ),
    }


@ttl_cache()
def get_communication_timing(customer_id: str) -> dict:
    from backend.capabilities import communication_timing as mod
    res = mod.recommend_communication(customer_id)

    import pandas as pd
    browsing = mod.browsing
    events   = browsing[browsing["customer_id"] == customer_id]
    if len(events) > 0:
        hourly_counts = events["hour"].value_counts().to_dict()
        hourly_pct = {
            str(h): float(count / len(events))
            for h, count in hourly_counts.items()
        }
    else:
        hourly_pct = {str(h): 0.1 for h in range(12, 19)}

    best_h = res.get("best_hour", 17)
    return {
        "best_send_hour_label":  f"{best_h:02d}:00",
        "best_day_of_week":      res.get("best_day", "Saturday"),
        "recommended_channel":   res.get("channel", "Email"),
        "predicted_open_rate":   res.get("open_rate", 0.22),
        "hourly_activity":       hourly_pct,
    }
