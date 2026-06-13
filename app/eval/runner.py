from __future__ import annotations

from typing import Any

from ..agent.graph import run_agent
from ..pipeline import ingest_ticker
from ..rag.retrieve import retrieve
from . import judge as J
from .dataset import DATASET, needed_filings

# Latest report is held in memory; GET /eval/results returns it. Resets on
# restart — persisting eval history is a reasonable follow-up.
_latest_report: dict[str, Any] | None = None


def latest_report() -> dict[str, Any] | None:
    return _latest_report


async def run_eval() -> dict[str, Any]:
    """Ingest the needed filings (idempotent), run each case, judge, aggregate."""
    global _latest_report

    for ticker, forms in needed_filings():
        await ingest_ticker(ticker, forms, limit=1)

    cases: list[dict[str, Any]] = []
    sums: dict[str, int] = {}
    counts: dict[str, int] = {}

    for case in DATASET:
        passage_objs = await retrieve(case.question, case.ticker, case.form_type)
        passages = [c.__dict__ for c in passage_objs]

        result = await run_agent(case.question, case.ticker, case.form_type)
        answer = result["answer"]
        cited = result["citations"] or passages

        metrics: dict[str, J.Verdict] = {}
        if case.kind == "guardrail":
            metrics["guardrail"] = await J.judge_guardrail(case.question, answer)
        else:
            metrics["retrieval_relevance"] = await J.judge_retrieval_relevance(
                case.question, passages
            )
            metrics["faithfulness"] = await J.judge_faithfulness(answer, cited)
            metrics["correctness"] = await J.judge_correctness(
                case.question, answer, case.expected_points
            )

        for name, verdict in metrics.items():
            sums[name] = sums.get(name, 0) + verdict.score
            counts[name] = counts.get(name, 0) + 1

        cases.append(
            {
                "question": case.question,
                "ticker": case.ticker,
                "kind": case.kind,
                "answer_preview": answer[:300],
                "n_citations": len(result["citations"]),
                "metrics": {
                    name: {"score": v.score, "reason": v.reason}
                    for name, v in metrics.items()
                },
            }
        )

    summary = {
        "n_cases": len(DATASET),
        "metrics": {
            name: {
                "mean_1to5": round(sums[name] / counts[name], 2),
                "normalized_0to1": round(sums[name] / counts[name] / 5, 3),
                "n": counts[name],
            }
            for name in sums
        },
    }

    _latest_report = {"summary": summary, "cases": cases}
    return _latest_report
