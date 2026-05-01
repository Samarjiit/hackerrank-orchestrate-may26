from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    issue: str
    subject: str = ""
    company: str = "None"

    def normalized_company(self) -> Optional[str]:
        c = self.company.strip().lower()
        if c in ("none", "", "null", "nan"):
            return None
        if "hackerrank" in c:
            return "hackerrank"
        if "claude" in c:
            return "claude"
        if "visa" in c:
            return "visa"
        return None

    def full_text(self) -> str:
        parts = []
        if self.subject and self.subject.strip():
            parts.append(f"Subject: {self.subject.strip()}")
        parts.append(self.issue.strip())
        return "\n".join(parts)


class TicketOutput(BaseModel):
    status: Literal["replied", "escalated"]
    product_area: str
    response: str
    justification: str
    request_type: Literal["product_issue", "feature_request", "bug", "invalid"]


class RetrievedDoc(BaseModel):
    content: str
    source: str
    company: str
    category: str
    title: str
    score: float = 0.0


class SafetyResult(BaseModel):
    is_injection: bool = False
    needs_escalation: bool = False
    escalation_reason: str = ""
    risk_level: Literal["low", "medium", "high"] = "low"


class TriageResult(BaseModel):
    ticket: TicketInput
    output: TicketOutput
    retrieved_docs: list[RetrievedDoc] = Field(default_factory=list)
    safety: SafetyResult = Field(default_factory=SafetyResult)
    inferred_company: Optional[str] = None
