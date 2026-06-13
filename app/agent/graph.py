from __future__ import annotations

from typing import Any, AsyncIterator

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from ..config import get_settings
from ..observability import ms, new_trace_id, now, recorder
from ..rag.retrieve import Citation
from .tools import TOOL_MAP, TOOLS, citation_collector

SYSTEM_PROMPT = """You are a research assistant for U.S. SEC filings (10-K, 10-Q, 8-K).

How to work:
- Ground every factual claim about a company in its filings by calling \
`retrieve_filings`. Cite the passages you use with their [n] markers, inline.
- Use `get_market_data` for current price / day range / market cap (live data, \
not from filings).
- Use `calculator` for ALL arithmetic (growth %, ratios, sums) — never compute \
in your head.
- If retrieval returns nothing relevant, say the filing may not be ingested yet \
rather than answering from memory.
- Be concise and specific. Quote or closely paraphrase the filing, and attach \
[n] citations to filing-derived statements.

GUARDRAIL — informational only, NOT investment advice:
- Do NOT give buy/sell/hold recommendations, price targets, or return predictions.
- If asked whether to buy/sell or for a price target, do NOT just refuse — instead \
give a balanced, filing-grounded analysis: lay out the potential strengths / \
tailwinds AND the risks / headwinds the filings and market data point to (a "bull \
case / bear case"), clearly labeled as informational considerations, and note that \
the decision is the reader's own.
- End answers that draw on filings or market data with: \
"This is informational analysis, not investment advice."
"""

MAX_ITERATIONS = 6


def _build_model() -> Any:
    s = get_settings()
    model = ChatAnthropic(
        model=s.llm_model,
        api_key=s.anthropic_api_key,
        max_tokens=2048,
        timeout=90,
    )
    return model.bind_tools(TOOLS)


def _chunk_text(content: Any) -> str:
    """Extract streamable text from a message chunk's content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _message_text(message: BaseMessage) -> str:
    return _chunk_text(message.content)


async def _run_tool(tool_call: dict[str, Any]) -> str:
    tool = TOOL_MAP.get(tool_call["name"])
    if tool is None:
        return f"Error: unknown tool {tool_call['name']!r}"
    try:
        return await tool.ainvoke(tool_call["args"])
    except Exception as exc:  # keep the loop alive; let the model react
        return f"Error running {tool_call['name']}: {exc}"


def _user_message(question: str, ticker: str | None, form_type: str | None) -> str:
    scope = []
    if ticker:
        scope.append(f"ticker {ticker.upper()}")
    if form_type:
        scope.append(f"form {form_type.upper()}")
    hint = f"  (Focus on {', '.join(scope)} where relevant.)" if scope else ""
    return f"{question}{hint}"


async def run_agent_events(
    question: str, ticker: str | None = None, form_type: str | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Drive the tool-calling loop, yielding events for SSE streaming.

    Event types:
      {"type": "token", "text": str}            incremental answer text
      {"type": "tool_start", "name", "args"}    a tool is about to run
      {"type": "tool_end", "name"}              a tool finished
      {"type": "final", "answer", "citations"}  terminal event
    """
    collector: list[Citation] = []
    token = citation_collector.set(collector)
    trace_id = new_trace_id()
    model_name = get_settings().llm_model
    try:
        model = _build_model()
        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(_user_message(question, ticker, form_type)),
        ]
        answer = ""

        for _ in range(MAX_ITERATIONS):
            acc: AIMessageChunk | None = None
            turn_start = now()
            async for chunk in model.astream(messages):
                acc = chunk if acc is None else acc + chunk
                text = _chunk_text(chunk.content)
                if text:
                    yield {"type": "token", "text": text}

            if acc is None:  # nothing streamed; bail out safely
                break
            recorder.record_llm(
                trace_id, "agent", model_name, acc.usage_metadata, ms(turn_start)
            )
            messages.append(acc)

            tool_calls = acc.tool_calls or []
            if not tool_calls:
                answer = _message_text(acc)
                break

            for tc in tool_calls:
                yield {"type": "tool_start", "name": tc["name"], "args": tc["args"]}
                tool_start = now()
                result = await _run_tool(tc)
                recorder.record_tool(
                    trace_id, tc["name"], ms(tool_start), not result.startswith("Error")
                )
                yield {"type": "tool_end", "name": tc["name"]}
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        if not answer and messages:
            answer = _message_text(messages[-1])

        yield {
            "type": "final",
            "answer": answer,
            "citations": [c.__dict__ for c in collector],
        }
    finally:
        citation_collector.reset(token)


async def run_agent(
    question: str, ticker: str | None = None, form_type: str | None = None
) -> dict[str, Any]:
    """Non-streaming convenience wrapper: returns {"answer", "citations"}."""
    final: dict[str, Any] = {"answer": "", "citations": []}
    async for event in run_agent_events(question, ticker, form_type):
        if event["type"] == "final":
            final = {"answer": event["answer"], "citations": event["citations"]}
    return final
