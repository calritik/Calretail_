"""
Module 4 — Customer Support Intelligence service layer.

Only voice-of-customer is memoised. The chatbot, triage and agent-assist
endpoints are conversational and may route through an LLM, so a cached reply
would be wrong rather than merely stale.
"""

from backend.utils.cache import ttl_cache
from typing import Optional

from backend.utils import naming

def chatbot_respond(customer_id: str, message: str, session_id: str) -> dict:
    from backend.capabilities import ai_chatbot as mod
    res = mod.chatbot_response(customer_id, message, session_id)
    res["session_id"] = session_id
    # Every card that shows a conversation also shows who it is with; without
    # this the header falls back to the raw id.
    res["customer_name"] = naming.customer(customer_id)
    if "powered_by" not in res:
        res["powered_by"] = "Notebook Engine"
    return naming.annotate(res)

def triage_ticket(description: str, customer_id: str) -> dict:
    from backend.capabilities import ticket_triage as mod
    res = mod.triage_ticket(description)
    res["customer_id"] = customer_id
    res["customer_name"] = naming.customer(customer_id)
    return naming.annotate(res)

def agent_assist(query_text: str, customer_id: str) -> dict:
    from backend.capabilities import agent_assist as mod
    res = mod.get_agent_assist(query_text)
    res["customer_id"] = customer_id
    res["customer_name"] = naming.customer(customer_id)
    return naming.annotate(res)

@ttl_cache()
def voice_of_customer(product_id: Optional[str] = None,
                       date_from: Optional[str] = None,
                       date_to: Optional[str] = None) -> dict:
    from backend.capabilities import voice_of_customer as mod
    res = mod.mine_customer_reviews(product_id, date_from, date_to)
    
    if not isinstance(res, list):
        res = []
        
    import pandas as pd
    df = pd.DataFrame(res)
    if df.empty:
        return {
            "total_reviews": 0, "avg_rating": 0.0, "alert": False,
            "sentiment_distribution": {}, "aspect_analysis": {}, "monthly_trend": []
        }
        
    total_reviews = len(df)
    avg_rating = float(df["rating"].mean())
    alert = avg_rating < 3.5
    
    # Sentiment distribution
    sent_counts = df["sentiment"].value_counts().to_dict()
    sentiment_distribution = {k: int(v) for k, v in sent_counts.items()}
    
    # Aspect analysis
    aspect_analysis = {}
    for aspect_name, group in df.groupby("aspect"):
        pos_count = sum(group["sentiment"] == "Positive")
        pct_pos = float(pos_count / len(group) * 100) if len(group) > 0 else 0.0
        aspect_analysis[aspect_name] = {
            "mention_count": int(len(group)),
            "avg_rating": float(group["rating"].mean()),
            "pct_positive": pct_pos
        }
        
    # Monthly trend — real average rating per calendar month from each
    # review's actual review_date, not fabricated placeholder months.
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    monthly = (
        df.dropna(subset=["review_date"])
        .assign(month_key=lambda d: d["review_date"].dt.to_period("M"))
        .groupby("month_key")["rating"].mean()
        .sort_index()
    )
    monthly_trend = [
        {"month": period.strftime("%b %Y"), "avg_rating": round(float(val), 2)}
        for period, val in monthly.items()
    ]
    
    return {
        "total_reviews": total_reviews,
        "avg_rating": round(avg_rating, 2),
        "alert": alert,
        "sentiment_distribution": sentiment_distribution,
        "aspect_analysis": aspect_analysis,
        "monthly_trend": monthly_trend
    }
