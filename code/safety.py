from __future__ import annotations

import re
from schemas import SafetyResult

# ── Prompt injection / jailbreak patterns ──────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(previous|all|above|prior)\s+(instructions?|prompts?|rules?|context)",
        r"forget\s+(everything|all|your|previous|prior)",
        r"disregard\s+(all|previous|your|prior|above)",
        r"override\s+(safety|guidelines?|rules?|instructions?|prompts?)",
        r"you\s+are\s+now\s+(a|an|the)",
        r"pretend\s+(you|that|to\s+be)",
        r"act\s+as\s+(if|though|a\b|an\b)",
        r"from\s+now\s+on\s+(you|ignore|forget|act)",
        r"new\s+instructions?\s*:",
        r"system\s*:\s*",
        r"<\s*/?system\s*>",
        r"\bDAN\b",
        r"jailbreak",
        r"(reveal|show|display|print|expose|output|dump)\s+(your|the)\s+(system\s*prompt|internal|hidden|secret|instructions?|rules?|retrieved|context)",
        r"(what\s+(is|are)|tell\s+me)\s+(your|the)\s+(system\s*prompt|internal\s+rules?|instructions?)",
        r"affiche\s+(toutes|les|tous)",
        r"(montre|affiche|donne\s+moi)\s+(les|tous|toutes)",
    ]
]

# ── High-risk escalation keywords ─────────────────────────────────────────────
# NOTE: Use possessive/active phrasing so FAQ questions ("how do I report a stolen card?")
# do NOT trigger escalation. Only active incident reports should trigger.
_ESCALATION_KEYWORDS: list[tuple[str, str]] = [
    # Fraud / financial crime — active incidents only
    ("i was defrauded", "Fraud reported"),
    ("i've been defrauded", "Fraud reported"),
    ("someone made unauthorized", "Unauthorized transaction reported"),
    ("unauthorized transaction on my", "Unauthorized transaction reported"),
    ("unauthorized charge on my", "Unauthorized charge reported"),
    ("chargeback", "Chargeback request — requires specialist"),
    ("payment dispute", "Payment dispute — requires specialist"),
    # Active theft reports — "my card was stolen", not FAQ questions about stolen cards
    ("my card was stolen", "Active card theft — escalate"),
    ("my card has been stolen", "Active card theft — escalate"),
    ("my card got stolen", "Active card theft — escalate"),
    ("my card is lost", "Card lost — escalate"),
    ("i lost my card", "Card lost — escalate"),
    ("i've lost my card", "Card lost — escalate"),
    # Identity / account security — active incidents
    ("identity theft", "Identity theft reported — immediate escalation required"),
    ("identity has been stolen", "Identity theft reported — immediate escalation required"),
    ("my identity was stolen", "Identity theft reported — immediate escalation required"),
    ("account hacked", "Account compromise reported"),
    ("account was hacked", "Account compromise reported"),
    ("account takeover", "Account takeover reported"),
    ("account compromised", "Account compromise reported"),
    ("compromised my account", "Account compromise reported"),
    ("someone accessed my account", "Unauthorized account access"),
    ("someone is using my account", "Account security concern"),
    # Legal threats
    ("legal action", "Legal threat — requires escalation"),
    ("file a lawsuit", "Legal threat — requires escalation"),
    ("take you to court", "Legal threat — requires escalation"),
    ("lawyer", "Legal threat — requires escalation"),
    ("attorney", "Legal threat — requires escalation"),
    ("file a complaint", "Formal complaint — requires escalation"),
    # Security incidents
    ("security vulnerability", "Security vulnerability report — requires escalation"),
    ("security flaw", "Security vulnerability report — requires escalation"),
    ("bug bounty", "Security report — requires specialist handling"),
    ("data breach", "Data breach concern — requires escalation"),
    # Billing / subscription requiring admin action
    ("pause our subscription", "Subscription change requiring account admin"),
    ("cancel our subscription", "Subscription change requiring account admin"),
    ("billing dispute", "Billing dispute — requires specialist"),
    # Site/service outages — infrastructure level, needs engineering
    ("site is down", "Site outage reported — requires engineering escalation"),
    ("website is down", "Site outage reported — requires engineering escalation"),
    ("none of the pages are accessible", "Site outage — requires escalation"),
    ("platform is down", "Platform outage — requires escalation"),
]

# ── Score manipulation / unfair advantage ─────────────────────────────────────
_INTEGRITY_KEYWORDS: list[tuple[str, str]] = [
    ("increase my score", "Score manipulation request — policy violation"),
    ("change my score", "Score manipulation request — policy violation"),
    ("modify my score", "Score manipulation request — policy violation"),
    ("graded me unfairly", "Score dispute — requires human review"),
    ("tell the company to", "Third-party intervention request — out of scope"),
    ("move me to the next round", "Recruitment outcome manipulation — escalate"),
]

# ── Out-of-scope patterns (still answered as invalid, not escalated) ───────────
_OOO_SCOPE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"(who|what)\s+is\s+the\s+(actor|director|singer|president|capital)",
        r"(iron\s*man|superman|batman|avengers)",
        r"delete\s+all\s+files",
        r"rm\s+-rf",
        r"format\s+(c:|the\s+hard\s+drive|my\s+computer)",
        r"give\s+me\s+(the\s+code\s+to|instructions?\s+to\s+)?(hack|exploit|attack)",
    ]
]


def check_safety(issue: str, subject: str = "") -> SafetyResult:
    """Run all safety checks and return a SafetyResult."""
    text = f"{subject}\n{issue}".lower()
    raw_text = f"{subject}\n{issue}"

    # 1. Prompt injection check
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(raw_text):
            return SafetyResult(
                is_injection=True,
                needs_escalation=True,
                escalation_reason="Prompt injection or jailbreak attempt detected in ticket.",
                risk_level="high",
            )

    # 2. High-risk escalation keywords
    for keyword, reason in _ESCALATION_KEYWORDS:
        if keyword.lower() in text:
            risk = "high" if any(
                k in keyword for k in ("identity theft", "fraud", "hacked", "takeover", "legal")
            ) else "medium"
            return SafetyResult(
                is_injection=False,
                needs_escalation=True,
                escalation_reason=reason,
                risk_level=risk,
            )

    # 3. Score/integrity issues
    for keyword, reason in _INTEGRITY_KEYWORDS:
        if keyword.lower() in text:
            return SafetyResult(
                is_injection=False,
                needs_escalation=True,
                escalation_reason=reason,
                risk_level="medium",
            )

    # 4. Clearly out-of-scope (low risk, no escalation — LLM will mark invalid)
    for pattern in _OOO_SCOPE_PATTERNS:
        if pattern.search(raw_text):
            return SafetyResult(
                is_injection=False,
                needs_escalation=False,
                escalation_reason="",
                risk_level="low",
            )

    return SafetyResult(risk_level="low")


def infer_company_from_text(issue: str, subject: str = "") -> str | None:
    """Heuristically infer the company from issue content when company=None."""
    text = f"{subject} {issue}".lower()

    hackerrank_signals = [
        "hackerrank", "test", "assessment", "candidate", "recruiter", "coding challenge",
        "interview", "hiring", "mock interview", "hackos", "screen", "skillup",
        "proctoring", "test variant", "question", "submission", "code challenge",
    ]
    claude_signals = [
        "claude", "anthropic", "conversation", "chat history", "claude.ai",
        "ai assistant", "bedrock", "lti", "claude for", "api key", "claude api",
        "claude desktop", "claude mobile", "claude code", "claude team",
    ]
    visa_signals = [
        "visa", "card", "payment", "merchant", "transaction", "atm", "cash",
        "debit", "credit card", "visa card", "bank", "charge", "refund",
        "traveller", "cheque", "travel", "pos", "contactless",
    ]

    hr_score = sum(1 for s in hackerrank_signals if s in text)
    cl_score = sum(1 for s in claude_signals if s in text)
    vi_score = sum(1 for s in visa_signals if s in text)

    scores = {"hackerrank": hr_score, "claude": cl_score, "visa": vi_score}
    best = max(scores, key=lambda k: scores[k])

    if scores[best] == 0:
        return None
    if list(scores.values()).count(scores[best]) > 1:
        return None
    return best
