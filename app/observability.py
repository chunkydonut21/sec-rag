from __future__ import annotations

import time
import uuid
from collections import deque
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

# USD per 1M tokens (input, output). Keep in sync with current Claude pricing.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _cost(model: str, input_tokens: int, output_tokens: int) -> float:
    base = next((m for m in PRICES if model.startswith(m)), None)
    if base is None:
        return 0.0
    p_in, p_out = PRICES[base]
    return round(input_tokens / 1e6 * p_in + output_tokens / 1e6 * p_out, 6)


@dataclass
class LLMCall:
    trace_id: str
    kind: str  # "agent" | "judge"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@dataclass
class ToolCall:
    trace_id: str
    name: str
    latency_ms: float
    ok: bool


# Correlates all calls within one request/agent run.
current_trace: ContextVar[str] = ContextVar("current_trace", default="-")


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:8]
    current_trace.set(tid)
    return tid


# Simple monotonic timing helpers (app code — not a workflow script).
def now() -> float:
    return time.perf_counter()


def ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


class Recorder:
    """In-memory ring buffer of recent LLM/tool calls plus rolling aggregates."""

    def __init__(self, maxlen: int = 500) -> None:
        self.llm: deque[LLMCall] = deque(maxlen=maxlen)
        self.tool: deque[ToolCall] = deque(maxlen=maxlen)

    def record_llm(
        self,
        trace_id: str,
        kind: str,
        model: str,
        usage: dict[str, Any] | None,
        latency_ms: float,
    ) -> None:
        usage = usage or {}
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        self.llm.append(
            LLMCall(
                trace_id=trace_id,
                kind=kind,
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=_cost(model, in_tok, out_tok),
                latency_ms=latency_ms,
            )
        )

    def record_tool(self, trace_id: str, name: str, latency_ms: float, ok: bool) -> None:
        self.tool.append(ToolCall(trace_id, name, latency_ms, ok))

    def summary(self) -> dict[str, Any]:
        llm = list(self.llm)
        tools = list(self.tool)

        by_tool: dict[str, list[float]] = {}
        for t in tools:
            by_tool.setdefault(t.name, []).append(t.latency_ms)

        return {
            "llm_calls": len(llm),
            "total_input_tokens": sum(c.input_tokens for c in llm),
            "total_output_tokens": sum(c.output_tokens for c in llm),
            "total_cost_usd": round(sum(c.cost_usd for c in llm), 6),
            "avg_llm_latency_ms": round(mean(c.latency_ms for c in llm), 1) if llm else 0,
            "tool_calls": len(tools),
            "tools": {
                name: {"count": len(lats), "avg_latency_ms": round(mean(lats), 1)}
                for name, lats in by_tool.items()
            },
        }

    def recent(self, n: int = 20) -> dict[str, Any]:
        return {
            "llm": [asdict(c) for c in list(self.llm)[-n:]],
            "tools": [asdict(t) for t in list(self.tool)[-n:]],
        }


recorder = Recorder()
