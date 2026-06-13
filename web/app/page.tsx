"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  type Citation,
  ingest as ingestApi,
  streamQuery,
} from "@/lib/api";

type Step = { name: string; args: Record<string, unknown>; done: boolean };
type Turn = {
  question: string;
  ticker?: string;
  form?: string;
  answer: string;
  steps: Step[];
  citations: Citation[];
  loading: boolean;
  error?: string;
};

const FORMS = ["", "10-K", "10-Q", "8-K"];

// Curated example questions, grouped so a first-time tester knows what to ask.
const EXAMPLE_GROUPS: { label: string; questions: string[] }[] = [
  {
    label: "Risks",
    questions: [
      "What are the main risk factors?",
      "Summarize the supply-chain and manufacturing risks.",
    ],
  },
  {
    label: "Business",
    questions: [
      "What are the company's main products and reportable segments?",
      "What does management say about competition in its markets?",
    ],
  },
  {
    label: "Markets + analysis",
    questions: [
      "What are the main risks, and what is the stock trading at now?",
      "What are the bull and bear considerations based on the filing?",
    ],
  },
];

export default function Home() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [ticker, setTicker] = useState("AAPL");
  const [form, setForm] = useState("10-K");

  const [ingestTicker, setIngestTicker] = useState("AAPL");
  const [ingestForm, setIngestForm] = useState("10-K");
  const [ingestStatus, setIngestStatus] = useState("");
  const [ingesting, setIngesting] = useState(false);

  const busy = turns.some((t) => t.loading);

  function patchTurn(idx: number, fn: (t: Turn) => Turn) {
    setTurns((prev) => {
      const copy = [...prev];
      if (copy[idx]) copy[idx] = fn(copy[idx]);
      return copy;
    });
  }

  async function ask(override?: string) {
    const q = (override ?? question).trim();
    if (!q || busy) return;
    const idx = turns.length;
    setTurns((prev) => [
      ...prev,
      {
        question: q,
        ticker: ticker || undefined,
        form: form || undefined,
        answer: "",
        steps: [],
        citations: [],
        loading: true,
      },
    ]);
    if (!override) setQuestion("");

    try {
      for await (const ev of streamQuery({
        question: q,
        ticker: ticker || null,
        form_type: form || null,
      })) {
        if (ev.type === "token") {
          patchTurn(idx, (t) => ({ ...t, answer: t.answer + ev.text }));
        } else if (ev.type === "tool_start") {
          patchTurn(idx, (t) => ({
            ...t,
            steps: [...t.steps, { name: ev.name, args: ev.args, done: false }],
          }));
        } else if (ev.type === "tool_end") {
          patchTurn(idx, (t) => {
            const steps = [...t.steps];
            for (let i = steps.length - 1; i >= 0; i--) {
              if (steps[i].name === ev.name && !steps[i].done) {
                steps[i] = { ...steps[i], done: true };
                break;
              }
            }
            return { ...t, steps };
          });
        } else if (ev.type === "final") {
          patchTurn(idx, (t) => ({
            ...t,
            answer: ev.answer || t.answer,
            citations: ev.citations,
            loading: false,
          }));
        }
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      patchTurn(idx, (t) => ({ ...t, loading: false, error: msg }));
    } finally {
      patchTurn(idx, (t) => ({ ...t, loading: false }));
    }
  }

  async function doIngest() {
    setIngesting(true);
    setIngestStatus("Ingesting… (fetching + embedding the filing — can take a few seconds)");
    try {
      const r = await ingestApi(ingestTicker.trim().toUpperCase(), ingestForm);
      let msg: string;
      if (r.filings_ingested > 0) {
        msg = `✓ Ingested ${r.filings_ingested} filing(s) · ${r.chunks_created} new chunks indexed. Ask a question below.`;
        if (r.filings_skipped > 0) {
          msg += ` (${r.filings_skipped} already on file.)`;
        }
      } else if (r.filings_skipped > 0) {
        msg = `✓ Already ingested — using ${r.chunks_existing} existing chunks. Ready to ask below.`;
      } else {
        msg = "No matching filings found for that ticker / form.";
      }
      setIngestStatus(msg);
    } catch (e) {
      setIngestStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setIngesting(false);
    }
  }

  return (
    <div className="wrap">
      <header className="app">
        <h1>SEC Filings Research Assistant</h1>
        <div className="disclaimer">
          Agentic RAG over SEC filings · Informational only — not investment advice.
        </div>
      </header>

      {/* Step 1 — ingest */}
      <div className="card toolbar">
        <span className="muted">
          <strong>1.</strong> Ingest filings:
        </span>
        <input
          className="ticker"
          value={ingestTicker}
          onChange={(e) => setIngestTicker(e.target.value)}
          placeholder="Ticker"
        />
        <select value={ingestForm} onChange={(e) => setIngestForm(e.target.value)}>
          {FORMS.filter(Boolean).map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
        <button className="ghost" onClick={doIngest} disabled={ingesting}>
          {ingesting ? "Ingesting…" : "Ingest"}
        </button>
        {ingestStatus && <span className="muted">{ingestStatus}</span>}
      </div>

      {/* Empty state — how to use + examples */}
      {turns.length === 0 && (
        <div className="intro">
          <p className="hint">
            <strong>2.</strong> Pick a ticker &amp; form below, then ask a question —
            or tap one of these to try it:
          </p>
          {EXAMPLE_GROUPS.map((g) => (
            <div className="examples" key={g.label}>
              <span className="examples-label">{g.label}</span>
              {g.questions.map((q) => (
                <button
                  key={q}
                  className="chip"
                  onClick={() => ask(q)}
                  disabled={busy}
                >
                  {q}
                </button>
              ))}
            </div>
          ))}
          <p className="muted small">
            Tip: the agent can combine tools — e.g. read the filing <em>and</em> fetch
            the live stock price in one answer. Filing facts are cited with [n] links
            to SEC.gov.
          </p>
        </div>
      )}

      {/* Conversation */}
      {turns.map((t, i) => (
        <div className="turn" key={i}>
          <div className="q">
            {t.question}
            {t.ticker && <span className="tag">{t.ticker}</span>}
            {t.form && <span className="tag">{t.form}</span>}
          </div>

          {t.steps.length > 0 && (
            <div className="steps">
              {t.steps.map((s, j) => (
                <div key={j} className={`step ${s.done ? "done" : "running"}`}>
                  <span className="name">{s.name}</span>{" "}
                  <span className="args">{JSON.stringify(s.args)}</span>
                </div>
              ))}
            </div>
          )}

          {t.answer ? (
            <div className="answer markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{t.answer}</ReactMarkdown>
            </div>
          ) : (
            t.loading && <div className="answer empty">Thinking…</div>
          )}

          {t.error && <div className="error">Error: {t.error}</div>}

          {t.citations.length > 0 && (
            <div className="citations">
              {t.citations.map((c, j) => (
                <a
                  className="citation"
                  key={j}
                  href={c.source_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  <div className="meta">
                    <span className="num">[{j + 1}]</span>
                    <span>
                      {c.ticker} {c.form_type} · {c.filing_date}
                    </span>
                    <span>· {c.section}</span>
                    <span>· score {c.score.toFixed(2)}</span>
                  </div>
                  <div className="snippet">{c.content.slice(0, 280)}…</div>
                </a>
              ))}
            </div>
          )}
        </div>
      ))}

      {/* Composer */}
      <div className="composer">
        <div className="inner">
          <div className="filters">
            <input
              className="ticker"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              placeholder="Ticker"
              title="Filter answers to this ticker"
            />
            <select
              value={form}
              onChange={(e) => setForm(e.target.value)}
              title="Filter answers to this filing type"
            >
              {FORMS.map((f) => (
                <option key={f || "any"} value={f}>
                  {f || "Any form"}
                </option>
              ))}
            </select>
          </div>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask();
              }
            }}
            placeholder="Ask about the filings… (Enter to send, Shift+Enter for newline)"
          />
          <button onClick={() => ask()} disabled={busy || !question.trim()}>
            {busy ? "…" : "Ask"}
          </button>
        </div>
      </div>
    </div>
  );
}
