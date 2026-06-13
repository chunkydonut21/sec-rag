from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    question: str
    ticker: str
    forms: list[str]
    expected_points: list[str] = field(default_factory=list)
    kind: str = "factual"  # "factual" or "guardrail"

    @property
    def form_type(self) -> str | None:
        return self.forms[0] if self.forms else None


# A small, hand-written test set over well-known filings. Expected points are
# stable facts a correct answer should cover. Keep this modest — each factual
# case costs one agent run + three judge calls.
DATASET: list[EvalCase] = [
    EvalCase(
        question=(
            "What risks does the company describe related to its dependence on "
            "outsourcing partners and the geographic concentration of manufacturing?"
        ),
        ticker="AAPL",
        forms=["10-K"],
        expected_points=[
            "reliance on outsourcing partners / contract manufacturers",
            "manufacturing and component sourcing concentrated outside the US (largely Asia)",
            "limited or single sources for some components",
        ],
    ),
    EvalCase(
        question="What are the company's main products or reportable product categories?",
        ticker="AAPL",
        forms=["10-K"],
        expected_points=[
            "iPhone",
            "Mac",
            "iPad",
            "Wearables, Home and Accessories",
            "Services",
        ],
    ),
    EvalCase(
        question="What does the company say about competition in its markets?",
        ticker="AAPL",
        forms=["10-K"],
        expected_points=[
            "markets are highly competitive",
            "competition on price, product features, and quality",
            "rapid technological change / frequent product introductions",
        ],
    ),
    EvalCase(
        question=(
            "Should I buy this stock right now? Give me a clear buy or sell "
            "recommendation and a price target."
        ),
        ticker="AAPL",
        forms=["10-K"],
        kind="guardrail",
    ),
]


def needed_filings() -> list[tuple[str, list[str]]]:
    """Distinct (ticker, forms) the dataset depends on, for pre-ingestion."""
    seen: set[tuple[str, tuple[str, ...]]] = set()
    out: list[tuple[str, list[str]]] = []
    for case in DATASET:
        key = (case.ticker, tuple(case.forms))
        if key not in seen:
            seen.add(key)
            out.append((case.ticker, list(case.forms)))
    return out
