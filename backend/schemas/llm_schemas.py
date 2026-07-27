"""
Pydantic schemas for structured LLM outputs (used with LangChain's
`PydanticOutputParser` via `backend.utils.llm_service.llm_structured` /
`parse_structured`). Keeping these here — rather than inline in the
notebooks — means the same schema drives prompting, parsing, and validation
for every provider (Groq/OpenAI/Gemini/Mock) consistently.
"""
from typing import Optional
from pydantic import BaseModel, Field


class ChatbotReply(BaseModel):
    """Structured reply from the Nexa support chatbot (notebook 13)."""
    response: str = Field(description="Natural-language reply to the customer, 1-3 sentences")
    intent: str = Field(default="general", description="order_status | return_status | complaint | general")
    escalate: bool = Field(default=False, description="Whether this needs human agent escalation")


class BuyingAssistantExtraction(BaseModel):
    """Structured intent/entity extraction for the conversational buying assistant (notebook 02)."""
    intent: str = Field(default="browse", description="buy | browse | compare | budget")
    category: Optional[str] = None
    brand: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    reply: Optional[str] = Field(default=None, description="Short, friendly one-sentence reply")


class AssistantRouterAction(BaseModel):
    """Structured routing decision for the top-level AI assistant router."""
    action: str = Field(default="general_chat")
    customer_id: Optional[str] = None
    product_id: Optional[str] = None
    response: Optional[str] = Field(default=None, description="Friendly reply if action is general_chat")
