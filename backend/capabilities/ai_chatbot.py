"""
CalRetail — 24x7 AI chatbot.

Ported from ``notebooks/capabilities/13_ai_chatbot.ipynb``. The notebook remains the readable
narrative of the method; this module is what the API actually runs.

State is built lazily by :func:`_init` on the first call, so importing this
module is free and nothing is computed for a capability nobody asks for.
:func:`reset` drops it again, which is how the process stays inside a small
memory budget without re-executing a notebook.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import math
from backend.utils.db import load_table
import warnings

warnings.filterwarnings('ignore')
from backend.capabilities import _registry

_READY = False
_BUILDING = False


def _init() -> None:
    """
    Build this capability's shared frames. Idempotent and cheap once warm.

    The _BUILDING guard matters: helpers lifted out of the setup block call
    _init() like every other function, and the setup itself calls those helpers.
    Without the guard that is unbounded recursion. Re-entering during the build
    simply returns, which leaves the helper reading the partially-built state —
    exactly what it saw when these were sequential notebook cells.
    """
    global _READY, _BUILDING, orders, returns, cust, llm_chat_with_memory, parse_structured, ChatbotReply, CHATBOT_SYSTEM_TEMPLATE
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        orders = load_table('orders')
        returns = load_table('returns')
        cust = load_table('customers')

        print(f"Chatbot support models loaded. Orders: {len(orders)} | Returns: {len(returns)}")

        from backend.utils.llm_service import llm_chat_with_memory, parse_structured
        from backend.schemas.llm_schemas import ChatbotReply

        CHATBOT_SYSTEM_TEMPLATE = """You are "Nexa", CalRetail's friendly customer support chatbot.
        Answer the customer's question using ONLY the real account context below — never invent order
        IDs, dates, statuses or amounts that aren't given to you. If the context doesn't contain the
        answer, politely say you don't have that information and offer to escalate to a human agent.
        Use the conversation history to stay consistent across turns.

        Customer ID: {customer_id}
        Customer name: {name}
        Recent orders: {orders_context}
        Recent returns: {returns_context}

        Return ONLY a JSON object with this exact shape:
        {{
          "response": "<your natural-language reply to the customer, 1-3 sentences>",
          "intent": "order_status" | "return_status" | "complaint" | "general",
          "escalate": true | false
        }}"""

        _READY = True
    finally:
        _BUILDING = False

    # Registering last bounds how many capabilities hold frames at once; the
    # coldest is reset when this one pushes the count over the limit.
    _registry.touch(__name__)


def __getattr__(name: str):
    """
    Build the state on first attribute access (PEP 562).

    Callers that reach past the public functions for a shared frame — the
    recommendations debug view reads the feedback matrix directly — would
    otherwise see an AttributeError, because nothing exists until _init() runs.
    This is only consulted for names *missing* from the module, so it costs
    nothing once warm.
    """
    if not name.startswith("__"):
        _init()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reset() -> None:
    """
    Release the cached frames so the next call rebuilds them.

    The names are *deleted*, not set to None. __getattr__ above only fires for
    names missing from the module, so leaving a None behind would hand a caller
    that None forever instead of triggering a rebuild — the frames would look
    released while every read of them silently broke.
    """
    global _READY
    _READY = False
    for _name in ('orders', 'returns', 'cust', 'llm_chat_with_memory', 'parse_structured', 'ChatbotReply', 'CHATBOT_SYSTEM_TEMPLATE'):
        globals().pop(_name, None)


def _pretty_date(value):
    """'2022-05-21 00:00:00' -> '21 May 2022'. Dates arrive as text from SQLite,
    and a customer-facing sentence should never contain a timestamp."""
    _init()
    ts = pd.to_datetime(value, errors='coerce')
    return value if pd.isna(ts) else ts.strftime('%d %b %Y')

def _rule_based_response(customer_id, question):
    """Deterministic fallback used when no LLM is configured, or the LLM
    output can't be parsed. Also used directly by the Mock LLM stand-in so
    the flow is fully exercised even without a provider API key."""
    _init()
    cust_orders = orders[orders['customer_id'] == customer_id].sort_values('order_date', ascending=False).head(3)
    cust_returns = returns[returns['customer_id'] == customer_id].sort_values('return_date', ascending=False).head(2)
    cname = cust[cust['customer_id'] == customer_id].iloc[0]['name'].split()[0] if customer_id in cust['customer_id'].values else "there"

    q_lower = question.lower()
    if 'order' in q_lower or 'delivery' in q_lower:
        if not cust_orders.empty:
            o = cust_orders.iloc[0]
            answer = f"Hi {cname}, your recent order {o['order_id']} is currently '{o['status']}'. It was ordered on {_pretty_date(o['order_date'])}."
        else:
            answer = f"Hi {cname}, I couldn't find any recent orders associated with your profile."
        intent = "order_status"
    elif 'return' in q_lower or 'refund' in q_lower:
        if not cust_returns.empty:
            r = cust_returns.iloc[0]
            answer = f"Hi {cname}, return request {r['return_id']} was received: status is {r['status'] or 'Processing'}."
        else:
            answer = f"Hi {cname}, you have no active return requests on file."
        intent = "return_status"
    else:
        answer = f"Hi {cname}! Welcome to CalRetail Support. How can I assist you with orders or returns today?"
        intent = "general"

    escalate = "complaint" in q_lower or "defect" in q_lower
    return {
        "response": answer,
        "intent": "complaint" if escalate else intent,
        "escalate": escalate,
    }


def chatbot_response(customer_id, question, session_id="default"):
    _init()
    cust_orders = orders[orders['customer_id'] == customer_id].sort_values('order_date', ascending=False).head(3)
    cust_returns = returns[returns['customer_id'] == customer_id].sort_values('return_date', ascending=False).head(2)
    cname = cust[cust['customer_id'] == customer_id].iloc[0]['name'].split()[0] if customer_id in cust['customer_id'].values else "there"

    orders_context = "; ".join(
        f"Order {o['order_id']}: {o['status']}, placed {_pretty_date(o['order_date'])}, total Rs.{o['total_amount']}"
        for _, o in cust_orders.iterrows()
    ) or "No recent orders on file."
    returns_context = "; ".join(
        f"Return {r['return_id']}: {r['status']}, reason {r['reason']}"
        for _, r in cust_returns.iterrows()
    ) or "No return requests on file."

    prompt = CHATBOT_SYSTEM_TEMPLATE.format(
        customer_id=customer_id, name=cname,
        orders_context=orders_context, returns_context=returns_context
    )
    # Real multi-turn memory: keyed per customer+session so the model actually sees prior
    # turns of THIS conversation (via LangChain's RunnableWithMessageHistory), rather than
    # `session_id` being an inert passthrough field on the response.
    memory_key = f"{customer_id}:{session_id}"
    raw = llm_chat_with_memory(memory_key, prompt, question, fallback="")
    result = parse_structured(raw, ChatbotReply)
    used_llm = result is not None

    if result is None:
        result = _rule_based_response(customer_id, question)
    else:
        result = result.model_dump()

    return {
        "customer_id": customer_id,
        "query": question,
        "response": result.get("response", ""),
        "intent": result.get("intent", "general"),
        "escalate": bool(result.get("escalate", False)),
        "powered_by": "LangChain LLM" if used_llm else "Rule-Based Engine",
    }
