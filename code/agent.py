from __future__ import annotations

import json
import time
from typing import Optional

import config
from prompts import (
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_TEMPLATE,
    TRIAGE_TOOL_SCHEMA,
    OPENAI_RESPONSE_SCHEMA,
    ESCALATION_RESPONSE,
    INVALID_RESPONSE,
    NO_CONTEXT_ESCALATION_RESPONSE,
)
from retriever import Retriever
from safety import check_safety, infer_company_from_text
from schemas import RetrievedDoc, SafetyResult, TicketInput, TicketOutput, TriageResult


class TriageAgent:
    """
    Orchestrates the full triage pipeline:
      safety check → company inference → retrieval → LLM classification → output
    """

    def __init__(self, retriever: Retriever, debug: bool = False) -> None:
        self._retriever = retriever
        self._debug = debug
        self._llm_client = self._build_llm_client()

    # ── Public API ─────────────────────────────────────────────────────────────

    def triage(self, ticket: TicketInput) -> TriageResult:
        """Process a single support ticket end-to-end."""

        # 1. Safety check
        safety = check_safety(ticket.issue, ticket.subject)

        # 2. Determine company
        company = ticket.normalized_company()
        inferred: Optional[str] = None
        if company is None:
            inferred = infer_company_from_text(ticket.issue, ticket.subject)
            company = inferred

        # 3. Immediate escalation for high-risk tickets — still retrieve for context
        docs = self._retriever.retrieve(ticket.full_text(), company=company)

        if safety.is_injection:
            output = TicketOutput(
                status="escalated",
                product_area=self._infer_product_area(docs, company),
                response=ESCALATION_RESPONSE,
                justification=f"Prompt injection detected: {safety.escalation_reason}",
                request_type="invalid",
            )
            return TriageResult(ticket=ticket, output=output, retrieved_docs=docs,
                                safety=safety, inferred_company=inferred)

        # 4. Call LLM for full triage
        output = self._llm_triage(ticket, company, docs, safety)

        return TriageResult(
            ticket=ticket,
            output=output,
            retrieved_docs=docs,
            safety=safety,
            inferred_company=inferred,
        )

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _llm_triage(
        self,
        ticket: TicketInput,
        company: Optional[str],
        docs: list[RetrievedDoc],
        safety: SafetyResult,
    ) -> TicketOutput:
        context = self._retriever.format_context(docs)
        company_display = (company or ticket.company or "Unknown").title()

        user_msg = TRIAGE_USER_TEMPLATE.format(
            company=company_display,
            subject=ticket.subject or "(none)",
            issue=ticket.issue,
            context=context,
            num_docs=len(docs),
        )

        # Prepend safety hint if needed
        if safety.needs_escalation:
            user_msg = (
                f"**[SAFETY FLAG]** {safety.escalation_reason}\n"
                f"This ticket has been flagged for likely escalation. "
                f"Still assess carefully but lean toward escalating.\n\n"
            ) + user_msg

        try:
            raw = self._call_llm(user_msg)
            output = TicketOutput(**raw)
            return output
        except Exception as exc:
            if self._debug:
                print(f"  [LLM error] {exc}")
            # Safe fallback
            return TicketOutput(
                status="escalated",
                product_area=self._infer_product_area(docs, company),
                response=NO_CONTEXT_ESCALATION_RESPONSE,
                justification=f"LLM call failed ({type(exc).__name__}). Escalated as safety measure.",
                request_type="product_issue",
            )

    def _call_llm(self, user_message: str) -> dict:
        provider = config.LLM_PROVIDER.lower()
        if provider == "anthropic":
            return self._call_anthropic(user_message)
        elif provider == "openai":
            return self._call_openai(user_message)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    def _call_anthropic(self, user_message: str) -> dict:
        import anthropic

        client: anthropic.Anthropic = self._llm_client
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=config.ANTHROPIC_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    system=TRIAGE_SYSTEM_PROMPT,
                    tools=[TRIAGE_TOOL_SCHEMA],
                    tool_choice={"type": "tool", "name": "triage_ticket"},
                    messages=[{"role": "user", "content": user_message}],
                )
                tool_use = next(b for b in response.content if b.type == "tool_use")
                return dict(tool_use.input)
            except Exception as exc:
                if attempt < max_retries - 1 and "overload" in str(exc).lower():
                    time.sleep(2 ** attempt)
                    continue
                raise

    def _call_openai(self, user_message: str) -> dict:
        from openai import OpenAI

        client: OpenAI = self._llm_client

        json_instruction = (
            "\n\nRespond with a single valid JSON object (no markdown, no code block) with exactly these keys: "
            '"status" (one of: replied, escalated), '
            '"product_area" (string), '
            '"response" (string), '
            '"justification" (string), '
            '"request_type" (one of: product_issue, feature_request, bug, invalid).'
        )

        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            seed=config.RANDOM_SEED,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message + json_instruction},
            ],
        )
        raw = response.choices[0].message.content
        return json.loads(raw)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _read_env_file() -> dict[str, str]:
        """Directly parse the .env file as a fallback, ignoring dotenv."""
        env: dict[str, str] = {}
        env_path = config.BASE_DIR / ".env"
        try:
            for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    env[key] = val
        except Exception:
            pass
        return env

    def _resolve_key(self, env_key: str) -> str:
        """Return env var from config, then .env file, then process env."""
        val = getattr(config, env_key, "")
        if not val:
            val = self._read_env_file().get(env_key, "")
        if not val:
            import os
            val = os.environ.get(env_key, "")
        return val

    def _build_llm_client(self):
        provider = config.LLM_PROVIDER.lower()
        if provider == "anthropic":
            api_key = self._resolve_key("ANTHROPIC_API_KEY")
            if not api_key or api_key == "your_anthropic_api_key_here":
                raise EnvironmentError("ANTHROPIC_API_KEY is not set. Check your .env file.")
            import anthropic
            return anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            api_key = self._resolve_key("OPENAI_API_KEY")
            if not api_key or api_key == "your_openai_api_key_here":
                raise EnvironmentError("OPENAI_API_KEY is not set. Check your .env file.")
            from openai import OpenAI
            return OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")

    @staticmethod
    def _infer_product_area(docs: list[RetrievedDoc], company: Optional[str]) -> str:
        if docs:
            return docs[0].category.split("/")[-1] or "general_support"
        if company:
            return f"{company}_support"
        return "general_support"
