"""
Every capability module builds and answers.

This replaces the old test_notebooks.py, which executed all sixteen notebooks
through the loader. That suite tested a path the application no longer takes —
the API imports these modules directly — and took minutes because each notebook
was re-executed cell by cell, demo charts included.
"""
import importlib

import pytest

# module -> (entry point, arguments, a key the result must carry)
CAPABILITIES = [
    ("personalised_recommendations", "get_recommendations", ("C00001", 5), "recommendations"),
    ("conversational_buying_assistant", "process_chat_message", ("C00001", "warm jacket under 3000"), "response"),
    ("next_best_offer", "resolve_nbo", ("C00001",), None),
    ("communication_timing", "recommend_communication", ("C00001",), None),
    ("demand_forecasting", "get_demand_forecast", ("P00001", 7), "forecast"),
    ("dynamic_pricing", "recommend_dynamic_price", ("P00001",), None),
    ("promotion_optimization", "analyze_promo_performance", ("PR000002",), None),
    ("competitor_price_monitoring", "detect_pricing_outliers", (), None),
    ("inventory_health_monitoring", "compute_inventory_health", (), None),
    ("automated_replenishment", "get_replenishment_parameters", ("P00001",), None),
    ("warehouse_slotting", "compute_abc_slotting_plan", ("W002",), None),
    ("route_optimisation", "solve_delivery_route", ("W002",), None),
    ("ai_chatbot", "chatbot_response", ("C00001", "where is my order", "s1"), "response"),
    ("ticket_triage", "triage_ticket", ("parcel arrived damaged, want refund",), None),
    ("agent_assist", "get_agent_assist", ("return window",), None),
    ("voice_of_customer", "mine_customer_reviews", (), None),
]


@pytest.mark.parametrize("mod_name,fn_name,args,required_key",
                         CAPABILITIES, ids=[c[0] for c in CAPABILITIES])
def test_capability_answers(mod_name, fn_name, args, required_key):
    mod = importlib.import_module(f"backend.capabilities.{mod_name}")

    fn = getattr(mod, fn_name, None)
    assert callable(fn), f"{mod_name} does not expose {fn_name}()"

    result = fn(*args)
    assert result is not None, f"{mod_name}.{fn_name} returned None"
    if isinstance(result, (list, dict)):
        assert len(result) > 0, f"{mod_name}.{fn_name} returned an empty result"
    if required_key:
        assert required_key in result, f"{mod_name}.{fn_name} is missing {required_key!r}"


@pytest.mark.parametrize("mod_name", [c[0] for c in CAPABILITIES])
def test_import_builds_nothing(mod_name):
    """
    Importing a capability must not do work.

    The whole point of the lazy build is that a process can import all sixteen
    and pay for none of them; if state leaked back to import time the memory
    budget would be spent before the first request.
    """
    mod = importlib.import_module(f"backend.capabilities.{mod_name}")
    mod.reset()
    assert mod._READY is False

    importlib.reload(mod)
    assert mod._READY is False, f"{mod_name} built state at import time"
