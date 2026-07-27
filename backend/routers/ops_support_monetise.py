"""
Module 3 — Operations Router  |  Module 4 — Support Router  |  Module 5 — Monetisation Router
"""
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Operations
# ─────────────────────────────────────────────────────────────────────────────
from backend.services.operations import (
    get_inventory_health, get_replenishment_order,
    optimise_warehouse, optimise_routes,
    get_markdown_candidates, get_inventory_timeseries,
)

ops_router = APIRouter(prefix="/api/v1/operations", tags=["Operational Excellence"])


class ReplenishRequest(BaseModel):
    product_id: str
    store_id: Optional[str] = None

class RouteRequest(BaseModel):
    warehouse_id: str
    order_ids: Optional[List[str]] = None


@ops_router.get("/inventory-health")
def inventory_health(
    store_id: Optional[str]    = Query(None),
    category: Optional[str]   = Query(None),
    top_n: int                 = Query(50, le=200),
):
    return {"results": get_inventory_health(store_id, category, top_n)}


@ops_router.post("/replenishment")
def replenishment(req: ReplenishRequest):
    result = get_replenishment_order(req.product_id, req.store_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@ops_router.get("/markdown-candidates")
def markdown_candidates(top_n: int = Query(8, ge=1, le=100)):
    """Overstocked SKUs ranked by the capital they are holding idle."""
    return get_markdown_candidates(top_n)


@ops_router.get("/inventory-timeseries")
def inventory_timeseries(product_id: str = Query(...),
                         days: int = Query(120, ge=7, le=1095)):
    """Daily units sold for one SKU, with the cover that pace implies."""
    return get_inventory_timeseries(product_id, days)


@ops_router.get("/warehouse-optimization")
def warehouse_optimization(warehouse_id: str = Query(...)):
    return optimise_warehouse(warehouse_id)


@ops_router.post("/route-optimization")
def route_optimization(req: RouteRequest):
    return optimise_routes(req.warehouse_id, req.order_ids)


@ops_router.get("/stores")
def list_stores(limit: int = Query(20, le=200)):
    from backend.utils.data_loader import get_stores
    df = get_stores()
    return df[["store_id","store_name","city","region","store_type"]].head(limit).to_dict(orient="records")


@ops_router.get("/warehouses")
def list_warehouses():
    from backend.utils.data_loader import get_warehouses
    df = get_warehouses()
    return df[["warehouse_id","warehouse_name","city","type"]].to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Support Intelligence
# ─────────────────────────────────────────────────────────────────────────────
from backend.services.support_intelligence import (
    chatbot_respond, triage_ticket, agent_assist, voice_of_customer,
)

support_router = APIRouter(prefix="/api/v1/support", tags=["Customer Support Intelligence"])


class ChatRequest(BaseModel):
    customer_id: str
    message: str
    session_id: str = "session-001"

class TriageRequest(BaseModel):
    ticket_description: str
    customer_id: str

class AgentRequest(BaseModel):
    query_text: str
    customer_id: str


@support_router.post("/chatbot")
def chatbot(req: ChatRequest):
    return chatbot_respond(req.customer_id, req.message, req.session_id)


@support_router.post("/ticket-triage")
def ticket_triage(req: TriageRequest):
    return triage_ticket(req.ticket_description, req.customer_id)


@support_router.post("/agent-assist")
def agent_assist_endpoint(req: AgentRequest):
    return agent_assist(req.query_text, req.customer_id)


class AssistantRouterRequest(BaseModel):
    query: str

@support_router.post("/assistant-router")
def assistant_router_endpoint(req: AssistantRouterRequest):
    """
    FastAPI Router endpoint to trigger LangChain-based AI routing logic.
    """
    from backend.utils.llm_service import llm_structured
    from backend.schemas.llm_schemas import AssistantRouterAction

    system_prompt = """You are a routing and extraction assistant.
Classify the user's retail platform query into one of these actions:
- "recommendations" (if asking for product suggestions/recommendations for a customer)
- "inventory_health" (if asking about stock levels/status/health/low stock)
- "demand_forecast" (if asking about demand forecasting or future product sales/MAPE)
- "ticket_triage" (if describing complaints, customer support tickets, or triaging issues)
- "buying_intent" (if assessing how likely a customer is to buy a specific product)
- "voice_of_customer" (if summarizing reviews, customer sentiment, or ratings)
- "general_chat" (any other greeting or general support question)

Also, extract customer_id (e.g. CUST-0123) and product_id (e.g. P-0456) if mentioned.

Return ONLY a JSON block:
{
  "action": "<one of the actions above>",
  "customer_id": "<CUST-XXXX or null>",
  "product_id": "<P-XXXX or null>",
  "response": "<friendly general response if action is general_chat, otherwise null>"
}
"""
    result = llm_structured(system_prompt, req.query, AssistantRouterAction)

    if result is None:
        # Deterministic keyword/regex safety net used when no LLM is configured, or the
        # LLM output can't be parsed/validated into AssistantRouterAction.
        import re
        action = "general_chat"
        q_low = req.query.lower()
        if "recommend" in q_low or "suggestion" in q_low:
            action = "recommendations"
        elif "inventory" in q_low or "stock" in q_low:
            action = "inventory_health"
        elif "forecast" in q_low or "demand" in q_low:
            action = "demand_forecast"
        elif "triage" in q_low or "ticket" in q_low or "complaint" in q_low:
            action = "ticket_triage"
        elif "buying intent" in q_low or "intent" in q_low:
            action = "buying_intent"
        elif "voice" in q_low or "review" in q_low or "sentiment" in q_low:
            action = "voice_of_customer"

        cid = None
        cm = re.search(r"cust[-_]?(\w+)", q_low)
        if cm:
            cid = f"CUST-{cm.group(1).upper()}"
        pid = None
        pm = re.search(r"p[-_]?(\d+)", q_low)
        if pm:
            pid = f"P-{pm.group(1).zfill(4)}"

        return {
            "action": action,
            "customer_id": cid,
            "product_id": pid,
            "response": "Hi there! How can I help you today?"
        }

    return result.model_dump()



@support_router.get("/voice-of-customer")
def voc(
    product_id: Optional[str] = Query(None),
    date_from:  Optional[str] = Query(None),
    date_to:    Optional[str] = Query(None),
):
    return voice_of_customer(product_id, date_from, date_to)


# (Module 5 — Monetisation removed: notebooks 17-20 not included in this platform deployment)
