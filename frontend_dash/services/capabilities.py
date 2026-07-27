"""
CalRetail — capability catalogue (16 capabilities, notebooks 01–16).

Every capability is backed by a Jupyter notebook in notebooks/capabilities/
and served live via FastAPI. Titles and qualifiers match the Calsoft deck.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Capability:
    key: str
    title: str
    blurb: str
    impact: str          # "Growth" | "Efficiency" | "Efficiency/Growth"
    data: str            # "High" | "Med" | "Low"
    speed: str           # "Fast" | "Med" | "Slow"
    wave: int | None = None
    source: str = "api"
    endpoint: str = ""

    @property
    def meta(self) -> dict:
        return {"impact": self.impact, "data": self.data, "speed": self.speed, "wave": self.wave}


@dataclass(frozen=True)
class Domain:
    key: str
    index: str
    title: str
    tagline: str
    summary: str
    path: str
    capabilities: list[Capability] = field(default_factory=list)


# ── Domain 01 — Customer Experience (notebooks 01–04) ────────────────────────
CUSTOMER_EXPERIENCE = Domain(
    key="cx", index="Domain 01",
    title="Customer Experience",
    tagline="Hyper-personalization & Discovery",
    summary=("The first and most visible AI battleground. Transforming seamless experiences "
             "across online, mobile, and physical stores to improve conversion and reduce friction."),
    path="/customer-experience",
    capabilities=[
        Capability("recommendations", "Hyper-personalized Recommendations",
                   "Collaborative-filtering recommendations driven by real purchase history.",
                   "Growth", "High", "Fast", 1, "api",
                   "POST /api/v1/customer-experience/recommendations"),
        Capability("assistant", "Personalized Buying Assistants",
                   "Conversational guidance to the right product.",
                   "Growth", "Med", "Med", None, "api",
                   "POST /api/v1/customer-experience/buying-assistant"),
        Capability("nbo", "Next-Best-Offer Engines",
                   "Triggers to improve conversion and basket size.",
                   "Growth", "High", "Fast", None, "api",
                   "GET /api/v1/customer-experience/next-best-offer"),
        Capability("comm_timing", "Communication Timing Optimiser",
                   "Best send time, day and channel per customer from real browsing patterns.",
                   "Efficiency", "High", "Fast", 2, "api",
                   "GET /api/v1/customer-experience/communication-timing"),
    ],
)

# ── Domain 02 — Merchandising (notebooks 05–08) ───────────────────────────────
MERCHANDISING = Domain(
    key="merch", index="Domain 02",
    title="Merchandising",
    tagline="Guiding Price, Assortment, and Placement.",
    summary=("AI-driven precision to balance demand, competition, and margin while optimizing "
             "inventory decisions across all regions and seasons."),
    path="/merchandising",
    capabilities=[
        Capability("forecast", "Demand Forecasting",
                   "Product and category-level inventory foresight.",
                   "Efficiency", "High", "Fast", 1, "api",
                   "GET /api/v1/merchandising/demand-forecast"),
        Capability("pricing", "Dynamic Pricing Engines",
                   "Balancing demand, competition, and margin in real time.",
                   "Growth", "High", "Fast", 1, "api",
                   "POST /api/v1/merchandising/dynamic-pricing"),
        Capability("promotion", "Promotion Optimization",
                   "Targeting offers by geography, persona, and buying signal.",
                   "Growth", "High", "Med", None, "api",
                   "GET /api/v1/merchandising/promotion-optimization"),
        Capability("competitor", "Competitor Price Monitoring",
                   "Intelligence across marketplaces and channels.",
                   "Efficiency", "Med", "Fast", None, "api",
                   "GET /api/v1/merchandising/competitor-monitoring"),
    ],
)

# ── Domain 03 — Operational Efficiency (notebooks 09–12) ─────────────────────
OPERATIONS = Domain(
    key="ops", index="Domain 03",
    title="Operational Efficiency",
    tagline="Orchestrating Inventory, Fulfillment, & Execution.",
    summary=("Smarter supply chain operations designed to reduce cost-to-serve, eliminate "
             "stock-outs, and streamline omnichannel order flows."),
    path="/operations",
    capabilities=[
        Capability("inventory", "Smart Inventory Management",
                   "Reducing stock-outs and overstocks across locations.",
                   "Efficiency", "Med", "Med", None, "api",
                   "GET /api/v1/operations/inventory-health"),
        Capability("replenishment", "Automated Replenishment",
                   "Responding to near-real-time demand patterns.",
                   "Efficiency", "High", "Med", 2, "api",
                   "POST /api/v1/operations/replenishment"),
        Capability("warehouse", "Warehouse Optimization",
                   "Improving picking, routing, and labor productivity.",
                   "Efficiency", "Low", "Slow", None, "api",
                   "GET /api/v1/operations/warehouse-optimization"),
        Capability("route", "Logistics, Route & Fleet Optimization",
                   "Real-time tracking and delivery efficiency.",
                   "Efficiency", "Med", "Fast", None, "api",
                   "POST /api/v1/operations/route-optimization"),
    ],
)

# ── Domain 04 — Customer Support (notebooks 13–16) ───────────────────────────
SUPPORT = Domain(
    key="support", index="Domain 04",
    title="Customer Support",
    tagline="The AI-Led Resolution Engine.",
    summary=("Shifting support from a cost center to an intelligent resolution layer that "
             "reduces triage time and up-skills human agents in real time."),
    path="/support",
    capabilities=[
        Capability("chatbot", "24x7 AI Chatbots",
                   "Support, order tracking, and product assistance.",
                   "Efficiency", "High", "Fast", 1, "api",
                   "POST /api/v1/support/chatbot"),
        Capability("triage", "Intelligent Ticket Triage",
                   "Routing to reduce resolution time and cost.",
                   "Efficiency", "High", "Fast", None, "api",
                   "POST /api/v1/support/ticket-triage"),
        Capability("agent_assist", "Agent Assist",
                   "Real-time resolution suggestions and SOP lookups for live agents.",
                   "Efficiency", "Med", "Fast", 2, "api",
                   "POST /api/v1/support/agent-assist"),
        Capability("voc", "Voice of Customer",
                   "Sentiment, aspect and trend mining from real product reviews.",
                   "Growth", "High", "Med", None, "api",
                   "GET /api/v1/support/voice-of-customer"),
    ],
)

DOMAINS = [CUSTOMER_EXPERIENCE, MERCHANDISING, OPERATIONS, SUPPORT]
BY_KEY = {d.key: d for d in DOMAINS}


def cap(domain_key: str, cap_key: str) -> Capability:
    """Looks up one capability — pages use this to pull a card's deck metadata."""
    for c in BY_KEY[domain_key].capabilities:
        if c.key == cap_key:
            return c
    raise KeyError(f"{domain_key}/{cap_key}")


def totals() -> dict:
    caps = [c for d in DOMAINS for c in d.capabilities]
    return {
        "domains": len(DOMAINS),
        "capabilities": len(caps),
        "live": sum(c.source == "api" for c in caps),
        "demo": sum(c.source == "demo" for c in caps),
        "wave1": sum(c.wave == 1 for c in caps),
        "wave2": sum(c.wave == 2 for c in caps),
        "wave3": sum(c.wave == 3 for c in caps),
    }
