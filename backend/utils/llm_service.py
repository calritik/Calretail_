"""
CalRetail — LLM Service (LangChain abstraction)
=================================================
Auto-detects the available LLM provider from .env:
  Priority: GROQ → OPENAI → GEMINI → rule-based fallback

Usage:
    from backend.utils.llm_service import get_llm, llm_chat, llm_available
    
    if llm_available():
        response = llm_chat([
            ("system", "You are a helpful retail assistant."),
            ("human", user_message)
        ])

Idiomatic LangChain helpers (prompt templates + LCEL chains + output parsers + memory):
    from backend.utils.llm_service import llm_structured, llm_chat_with_memory, parse_structured

    # Single-turn structured extraction via a `ChatPromptTemplate | llm | PydanticOutputParser` chain
    result = llm_structured(system_prompt, human_message, MySchema)   # -> MySchema | None

    # Multi-turn chat with real per-session memory via `RunnableWithMessageHistory`
    raw_text = llm_chat_with_memory(session_id, system_prompt, human_message, fallback="")
    parsed = parse_structured(raw_text, MySchema)                     # -> MySchema | None
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from backend.utils.logger import logger

T = TypeVar("T", bound=BaseModel)

# Load .env automatically
def _load_env():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip()
                    if val and key not in os.environ:
                        os.environ[key] = val

_load_env()


@lru_cache(maxsize=1)
def _detect_provider() -> str:
    """Auto-detect which LLM provider to use based on available API keys."""
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced and forced != "rule_based":
        return forced

    if os.getenv("GROQ_API_KEY", "").strip():
        return "groq"
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.getenv("GEMINI_API_KEY", "").strip():
        return "gemini"
    return "rule_based"


@lru_cache(maxsize=1)
def get_llm():
    """
    Returns a LangChain chat model instance, or None if no API key available.
    
    Returns:
        langchain BaseChatModel instance, or None
    """
    provider = _detect_provider()
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "512"))

    try:
        if provider == "groq":
            from langchain_groq import ChatGroq
            model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"LLM: Groq ({model_name}) ✅")
            return llm

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"LLM: OpenAI ({model_name}) ✅")
            return llm

        elif provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            # "gemini-1.5-flash" was retired by Google and now 404s; use the
            # self-updating "-latest" alias so this doesn't go stale again.
            model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
            llm = ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GEMINI_API_KEY"),
                model=model_name,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            logger.info(f"LLM: Gemini ({model_name}) ✅")
            return llm

        else:
            return MockCalRetailChatModel()

    except ImportError as e:
        logger.warning(f"LLM provider '{provider}' not installed: {e}. Using mock model.")
    except Exception as e:
        logger.warning(f"LLM init failed ({provider}): {e}. Using mock model.")

    return MockCalRetailChatModel()


from typing import Any, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration

class MockCalRetailChatModel(BaseChatModel):
    """
    A custom mock LangChain chat model that provides local fallback responses
    by mirroring rule-based results in JSON format or conversational text.
    """
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_msg = ""
        system_msg = ""
        for m in messages:
            if m.type == "human" or m.type == "user":
                user_msg = m.content
            elif m.type == "system":
                system_msg = m.content

        user_low = user_msg.lower()
        sys_low = system_msg.lower()
        import json

        # 1. AI Assistant Router logic
        if "routing and extraction assistant" in sys_low:
            action = "general_chat"
            response = "Hi there! How can I help you today?"
            cid = None
            pid = None
            
            import re
            if "recommend" in user_low or "suggestion" in user_low:
                action = "recommendations"
            elif "inventory" in user_low or "stock" in user_low:
                action = "inventory_health"
            elif "forecast" in user_low or "demand" in user_low:
                action = "demand_forecast"
            elif "triage" in user_low or "ticket" in user_low or "complaint" in user_low:
                action = "ticket_triage"
            elif "buying intent" in user_low or "intent" in user_low:
                action = "buying_intent"
            elif "voice" in user_low or "review" in user_low or "sentiment" in user_low:
                action = "voice_of_customer"
                
            cust_m = re.search(r"cust[-_]?(\w+)", user_low)
            if cust_m:
                cid = f"CUST-{cust_m.group(1).upper()}"
            prod_m = re.search(r"p[-_]?(\d+)", user_low)
            if prod_m:
                pid = f"P-{prod_m.group(1).zfill(4)}"
                
            resp_dict = {
                "action": action,
                "customer_id": cid,
                "product_id": pid,
                "response": response
            }
            content = json.dumps(resp_dict)

        # 2. Support Chatbot logic
        elif "customer_context" in sys_low or "escalate" in sys_low:
            import re
            # Extract the real customer ID from the ORIGINAL (non-lowercased)
            # system prompt so case is preserved (dataset IDs are like "C00001").
            cid = "C00001"
            cf = re.search(r"customer id:\s*(\S+)", system_msg, re.IGNORECASE)
            if cf:
                cid = cf.group(1).strip()

            from backend.capabilities import ai_chatbot as mod
            resp_dict = mod._rule_based_response(cid, user_msg)
            resp_dict["confidence"] = 0.85
            content = json.dumps(resp_dict)

        # 3. Ticket Triage Assistant logic
        elif "ticket triage assistant" in sys_low:
            cat = "Order Issue"
            prio = "Medium"
            if "wrong" in user_low or "someone else" in user_low:
                cat = "Wrong Item"
                prio = "High"
            elif "size" in user_low or "exchange" in user_low or "fit" in user_low:
                cat = "Size Exchange"
                prio = "Low"
            elif "delay" in user_low or "arrive" in user_low or "tracking" in user_low:
                cat = "Delivery Delay"
                prio = "Medium"
            elif "damage" in user_low or "broken" in user_low or "defect" in user_low or "quality" in user_low:
                cat = "Product Quality"
                prio = "High"
            elif "refund" in user_low or "return" in user_low:
                cat = "Return & Refund"
                prio = "Medium"
            elif "payment" in user_low or "charged" in user_low or "card" in user_low:
                cat = "Payment Problem"
                prio = "High"
            elif "account" in user_low or "login" in user_low:
                cat = "Account Issue"
                prio = "Low"
                
            resp_dict = {
                "category": cat,
                "priority": prio
            }
            content = json.dumps(resp_dict)

        # 4. Buying Assistant Agent logic
        elif "exact category" in sys_low:
            from backend.utils.data_loader import get_products
            from backend.capabilities import conversational_buying_assistant as mod
            extracted = mod._extract_intent_rules(user_msg, get_products())
            content = json.dumps(extracted)

        # 5. Fallback general text response
        else:
            content = f"Mock response to: {user_msg}"

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    @property
    def _llm_type(self) -> str:
        return "mock-calretail"


def llm_available() -> bool:
    """Returns True if an LLM is configured and importable."""
    return get_llm() is not None


def llm_chat(messages: list[tuple[str, str]], fallback: str = "") -> str:
    """
    Send a list of (role, content) messages to the LLM and return the text response.
    
    Args:
        messages: List of ("system"|"human"|"assistant", content) tuples
        fallback: String to return if LLM is unavailable or errors
        
    Returns:
        LLM response text, or fallback string
    """
    llm = get_llm()
    if llm is None:
        return fallback

    try:
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        lc_messages = []
        for role, content in messages:
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "human" or role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant" or role == "ai":
                lc_messages.append(AIMessage(content=content))

        response = llm.invoke(lc_messages)
        return response.content.strip()

    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Idiomatic LangChain layer: PromptTemplate + LCEL chains + output parsers + memory
# ─────────────────────────────────────────────────────────────────────────────

_session_histories: dict = {}


def _get_session_history(session_id: str):
    """Per-session in-memory chat history store, used by `RunnableWithMessageHistory`
    so multi-turn conversations (e.g. the support chatbot) actually remember prior turns
    instead of just threading an unused `session_id` string through the response."""
    from langchain_core.chat_history import InMemoryChatMessageHistory
    if session_id not in _session_histories:
        _session_histories[session_id] = InMemoryChatMessageHistory()
    return _session_histories[session_id]


def llm_structured(system_prompt: str, human_message: str, schema: Type[T]) -> Optional[T]:
    """
    Single-turn structured extraction using an idiomatic LCEL chain:
    `ChatPromptTemplate | chat_model | PydanticOutputParser`.

    Returns a validated instance of `schema`, or None if the LLM errors or its output
    can't be parsed/validated into `schema` — callers should fall back to a deterministic
    rule-based parser in that case (same contract as the old raw-string `llm_chat`).
    """
    llm = get_llm()
    if llm is None:
        return None
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import SystemMessage
        from langchain_core.output_parsers import PydanticOutputParser

        # `system_prompt` is already fully rendered by the caller (and may contain literal
        # `{...}` braces from JSON-shape examples), so pass it as a literal `SystemMessage`
        # rather than a ("system", text) template tuple — otherwise ChatPromptTemplate would
        # try to re-parse those braces as its own template variables.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("human", "{input}"),
        ])
        parser = PydanticOutputParser(pydantic_object=schema)
        chain = prompt | llm | parser
        return chain.invoke({"input": human_message})
    except Exception as e:
        logger.warning(f"LLM structured call failed ({schema.__name__}): {e}")
        return None


def parse_structured(raw_text: str, schema: Type[T]) -> Optional[T]:
    """Validate/parse a raw LLM text response into `schema` via `PydanticOutputParser`,
    tolerating markdown code-fenced JSON. Returns None on any failure."""
    if not raw_text:
        return None
    try:
        from langchain_core.output_parsers import PydanticOutputParser
        return PydanticOutputParser(pydantic_object=schema).parse(raw_text)
    except Exception:
        return None


def llm_chat_with_memory(session_id: str, system_prompt: str, human_message: str, fallback: str = "") -> str:
    """
    Multi-turn chat using LangChain's `RunnableWithMessageHistory`, keyed by `session_id`,
    so the model actually sees prior turns of THIS conversation (real memory), unlike a
    plain stateless `llm_chat()` call.
    """
    llm = get_llm()
    if llm is None:
        return fallback
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.messages import SystemMessage
        from langchain_core.runnables.history import RunnableWithMessageHistory

        # Same literal-braces consideration as `llm_structured` above.
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        chain = prompt | llm
        chain_with_history = RunnableWithMessageHistory(
            chain, _get_session_history,
            input_messages_key="input", history_messages_key="history",
        )
        response = chain_with_history.invoke(
            {"input": human_message},
            config={"configurable": {"session_id": session_id}},
        )
        return response.content.strip()
    except Exception as e:
        logger.warning(f"LLM memory chat failed: {e}")
        return fallback


def llm_provider_name() -> str:
    """Returns the active provider name for display in UI."""
    p = _detect_provider()
    models = {
        "groq":       os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "openai":     os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "gemini":     os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "rule_based": "Rule-Based Engine",
    }
    return f"{p.title()} / {models.get(p, '')}"
