from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.agent import SupportAgent


app = FastAPI(
    title="Amazon Support Agent",
    description=(
        "Customer support agent using intent classification, "
        "historical retrieval, grounded generation, "
        "and escalation."
    ),
    version="1.0.0"
)


_agent = None


class SupportRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        description="Customer support message"
    )


class SupportResponse(BaseModel):

    customer_message: str
    intent: str
    intent_reason: str
    reply: str
    grounding_note: str
    decision: str
    decision_reason: str
    evidence_ids: list[str]


def get_agent():

    global _agent

    if _agent is None:
        _agent = SupportAgent()

    return _agent


@app.get("/")
def root():

    return {
        "service": "Amazon Support Agent",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/support",
    response_model=SupportResponse
)
def support(request: SupportRequest):

    agent = get_agent()

    return agent.handle(
        request.message
    )