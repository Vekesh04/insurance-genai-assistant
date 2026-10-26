from __future__ import annotations

import re
from dataclasses import dataclass, asdict

CLAIM_RE = re.compile(r"\bCLM\d+\b", re.I)
POLICY_RE = re.compile(r"\bPOL\d+\b", re.I)
CUSTOMER_RE = re.compile(r"\bCUST\d+\b", re.I)
HOSPITAL_RE = re.compile(r"\bHOSP\d+\b", re.I)

POLICY_TERMS = {
    "cover", "covered", "coverage", "exclusion", "excluded", "waiting period",
    "maternity", "pre-existing", "reimbursement", "policy wording", "clause",
    "benefit", "deductible", "copay", "co-pay", "procedure", "eligible",
}
SQL_TERMS = {
    "status", "claim amount", "approved amount", "missing document", "documents",
    "customer", "hospital", "network", "fraud risk", "manual review", "premium",
    "sum insured", "policy status", "interaction",
}
COMBINED_TERMS = {
    "analyze claim", "analyse claim", "claim analysis", "why is my claim",
    "is this claim covered", "check this claim", "claim eligibility",
    "documents are missing", "documents missing", "should this claim",
}


@dataclass
class RouteDecision:
    route: str
    reason: str
    entities: dict

    def to_dict(self) -> dict:
        return asdict(self)


def extract_entities(question: str) -> dict:
    def one(pattern):
        match = pattern.search(question)
        return match.group(0).upper() if match else None

    return {
        "claim_id": one(CLAIM_RE),
        "policy_id": one(POLICY_RE),
        "customer_id": one(CUSTOMER_RE),
        "hospital_id": one(HOSPITAL_RE),
    }


def route_question(question: str) -> RouteDecision:
    q = question.lower().strip()
    entities = extract_entities(question)

    if any(term in q for term in COMBINED_TERMS) and entities["claim_id"]:
        return RouteDecision("combined", "Claim analysis requires structured claim data plus policy evidence.", entities)

    # A claim ID plus explicit coverage/eligibility language needs both sources.
    if entities["claim_id"] and any(term in q for term in {"covered", "coverage", "eligible", "exclusion", "waiting period"}):
        return RouteDecision("combined", "Question refers to a specific claim and a policy rule.", entities)

    if entities["claim_id"] or entities["policy_id"] or entities["customer_id"] or entities["hospital_id"]:
        if any(term in q for term in POLICY_TERMS) and not any(term in q for term in SQL_TERMS):
            return RouteDecision("rag", "Question asks for policy wording/rules.", entities)
        return RouteDecision("sql", "Question asks for a known business record or transaction fact.", entities)

    if any(term in q for term in POLICY_TERMS):
        return RouteDecision("rag", "Policy concepts are answered from policy documents.", entities)

    if any(term in q for term in SQL_TERMS):
        return RouteDecision("sql", "Operational facts are answered from the SQL database.", entities)

    return RouteDecision("general", "No strong structured or policy-document signal was detected.", entities)
