export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type Citation = {
  content: string;
  section: string;
  ticker: string;
  form_type: string;
  filing_date: string;
  accession: string;
  source_url: string;
  score: number;
};

export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "tool_start"; name: string; args: Record<string, unknown> }
  | { type: "tool_end"; name: string }
  | { type: "final"; answer: string; citations: Citation[] };

export type QueryBody = {
  question: string;
  ticker?: string | null;
  form_type?: string | null;
};

/**
 * Stream the agent's events from POST /query/stream. The backend sends SSE
 * frames ("event: <type>\ndata: <json>\n\n"); we read the response body and
 * parse frames, yielding the JSON `data` payload (which already carries `type`).
 */
export async function* streamQuery(body: QueryBody): AsyncGenerator<AgentEvent> {
  const res = await fetch(`${API_BASE}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok || !res.body) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line; tolerate both \n\n and \r\n\r\n.
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      const data = frame
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("");
      if (!data) continue; // skip keep-alive comments
      try {
        yield JSON.parse(data) as AgentEvent;
      } catch {
        /* ignore malformed frame */
      }
    }
  }
}

export type IngestResult = {
  ticker: string;
  filings_ingested: number;
  filings_skipped: number;
  chunks_created: number;
  chunks_existing: number;
};

export async function ingest(
  ticker: string,
  form: string,
  limit = 1,
): Promise<IngestResult> {
  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ticker, forms: [form], limit }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data as IngestResult;
}
