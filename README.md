# Prospectus — Multi-Agent Financial Due-Diligence System

**Type a ticker. Get a citation-grounded investment memo in under 30 seconds — every claim traceable to real SEC data.**

Investment due diligence normally takes an analyst 4–8 hours: reading 10-Ks, computing ratios by hand, checking sentiment, cross-referencing insider trades. Prospectus automates this with four specialist agents — Data, Quant, Sentiment, Risk — each exposed as an independent [MCP](https://modelcontextprotocol.io) server, orchestrated by a checkpointed **LangGraph** pipeline. A self-critique node verifies every claim in the final memo against real agent output before it's ever shown to a user, and routes a targeted retry to whichever agent produced an unsupported claim.

🔗 **Live demo:** [prospectus-multi-agent-financial-du.vercel.app](https://prospectus-multi-agent-financial-du.vercel.app)

---

## Why this is different

Most agent demos wire tools directly into one LangChain agent and call it done. Prospectus makes two different bets:

- **MCP-native, not app-locked.** Every agent's tools are typed, audit-logged, and independently callable — any MCP client (Claude Desktop, another LangGraph app) can call the Quant Agent's `compute_ratios` without touching this codebase.
- **Grounded, not just plausible.** Every claim in the memo carries a `source_field` — a dotted path into real agent state (`ratios.debt_to_equity`). A deterministic check (not another LLM call) verifies it resolves before the memo is accepted.

## Architecture

```mermaid
flowchart TD
    START([User submits ticker]) --> SUP[supervisor<br/>CIK resolution + agent routing]
    SUP --> DATA[data_agent<br/>SEC EDGAR: filings, XBRL, insider trades]
    DATA --> QUANT[quant_agent<br/>8 ratios + debt history]
    DATA --> SENT[sentiment_agent<br/>FinBERT on relevance-filtered news]
    QUANT --> RISK[risk_agent<br/>3 red-flag detectors]
    SENT --> MEMO
    RISK --> MEMO["memo_writer (deferred)<br/>GPT-4.1-mini structured synthesis"]
    MEMO --> CRITIC[self_critic<br/>deterministic grounding check]
    CRITIC -->|"citation coverage OK"| DONE([Memo returned via SSE])
    CRITIC -.->|"ungrounded claim found<br/>(retry_count < 2)"| RETRY[["targeted retry:<br/>data_agent / quant_agent /<br/>sentiment_agent / risk_agent /<br/>memo_writer"]]
    RETRY -.-> CRITIC

    style MEMO fill:#e8d5f2
    style CRITIC fill:#ffe8cc
    style RETRY fill:#ffd6d6
```

`memo_writer` is marked **deferred** deliberately — its two predecessors (`sentiment_agent`, `risk_agent`) sit at different distances from `data_agent`, and LangGraph's default fan-in behavior fires a node once per completed predecessor rather than once after all complete. Without `defer=True` this silently ran the memo synthesis twice per request.

## Message flow — one full request

```mermaid
sequenceDiagram
    participant U as Browser
    participant FE as React Frontend
    participant API as FastAPI (/diligence/stream)
    participant LG as LangGraph (Postgres-checkpointed)
    participant DA as Data Agent :9001
    participant QA as Quant Agent :9002
    participant SA as Sentiment Agent :9003
    participant RA as Risk Agent :9004
    participant LLM as GPT-4.1-mini

    U->>FE: Enter ticker, click Analyse
    FE->>API: GET /diligence/stream?ticker=NVDA
    API->>LG: graph.astream(config={thread_id: run_id})
    LG->>DA: resolve_cik, get_company_facts,<br/>get_filing_sections, get_insider_transactions
    DA-->>LG: filings, XBRL facts, insider data
    par fan-out
        LG->>QA: compute_ratios, get_ratio_history
        LG->>SA: get_sentiment_summary
    end
    QA-->>LG: 8 ratios + debt history
    SA-->>LG: FinBERT sentiment scores
    LG->>RA: run_all_checks(ratio_history, filings)
    RA-->>LG: risk flags
    LG->>LLM: synthesize memo (instructor, structured schema)
    LLM-->>LG: claims, each with a source_field citation
    LG->>LG: self_critic — verify every source_field resolves
    alt grounding fails
        LG->>LG: targeted retry (capped at 2)
    else grounded
        LG-->>API: final memo + citation_coverage
    end
    API-->>FE: SSE — trace_entry / state_update / end
    FE-->>U: live pipeline trace + citation-grounded memo
```

## Evaluation

Verified on an 8-ticker golden set (NVDA, AAPL, MSFT, GOOGL, BA, INTC, RIVN, SMCI), each entry hand-checked against real 10-K figures — not tuned to match whatever the pipeline happened to output.

| Metric | Result |
|---|---|
| Ratio accuracy | **100%** — every computed ratio within 2% of hand-verified figures |
| Avg. citation coverage | **100%** |
| Red-flag precision | **100%** |
| Red-flag recall | 87.5% (7/8) |
| Latency (p50 / p95) | ~24s / ~30s per ticker |

## Tech stack

`Python` · `LangGraph` · `MCP` · `GPT-4.1-mini` · `FinBERT` · `SEC EDGAR XBRL` · `PostgreSQL` · `FastAPI` · `React` + `Vite` + `Tailwind` · `Railway` · `Vercel`

## Quick start

```bash
# 1. Start Postgres
docker-compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Copy and fill env vars
cp .env.example .env

# 4. Run each MCP agent (separate terminals)
python -m mcp_servers.data_agent.server --transport http
python -m mcp_servers.quant_agent.server --transport http
python -m mcp_servers.sentiment_agent.server --transport http
python -m mcp_servers.risk_agent.server --transport http

# 5. Run the backend
uvicorn backend.api.main:app --reload

# 6. Run the frontend
cd frontend && npm run dev
```

Or run any single agent over stdio for Claude Desktop testing:

```bash
python -m mcp_servers.data_agent.server --transport stdio
```

## MCP servers

| Agent | Port | Responsibility |
|---|---|---|
| `data_agent` | 9001 | SEC EDGAR — filings, XBRL facts, insider Form 4 trades |
| `quant_agent` | 9002 | Ratio computation, peer comparison, debt-to-equity history |
| `sentiment_agent` | 9003 | Relevance-filtered news, FinBERT sentiment scoring |
| `risk_agent` | 9004 | Debt spikes, insider-selling clusters, audit-language flags |

In production (Railway), all four agents and the API run as one Docker image managed by `supervisord` — they talk to each other over `localhost`, which is both faster than a network hop and structurally guarantees they're never publicly reachable.

Each is a standalone `mcp.server.Server` served over streamable-HTTP, sharing one Postgres instance for caching and audit logging.

## Authentication — design decision

### What's here

The MCP servers use a **static bearer token** per agent (`MCP_DATA_AGENT_TOKEN`, etc.), checked by `_BearerAuthMiddleware` on every HTTP request:

```
MCP_DATA_AGENT_TOKEN=some-long-random-string
```

Clients send:

```
Authorization: Bearer some-long-random-string
```

For **stdio transport** (Claude Desktop local testing) there's no auth check — stdio is a local-process pipe, so network-level auth is moot.

### What production replaces this with

The MCP specification (§4.3) mandates **OAuth 2.1** for production deployments:

| Requirement | RFC / spec |
|---|---|
| Discovery endpoint | RFC 8414 `/.well-known/oauth-authorization-server` |
| PKCE-protected authorization code flow | RFC 7636 |
| Short-lived, scoped access tokens | e.g. `read:edgar`, `read:insider` |
| Token introspection on every request | RFC 7662 |

`_BearerAuthMiddleware` is the exact injection point for this swap — replace the static string comparison with a JWKS-verified JWT check or a token-introspection call, and nothing else in the server changes.

**This is a deliberate portfolio-scope decision, not an oversight.** Full OIDC plumbing would add ~300 lines of boilerplate that obscures the agent architecture being demonstrated. In a production engagement this would be handled by an API gateway (Kong, AWS API Gateway) in front of the MCP servers, so the server code stays identical.

## Database tables

| Table | Purpose | TTL |
|---|---|---|
| `ticker_cache` | SEC ticker → CIK map | 24h |
| `filing_cache` | Submissions + company-facts blobs | 24h |
| `document_cache` | Raw 10-K/10-Q HTML | none (immutable) |
| `form4_cache` | Form 4 XML | none (immutable) |
| `news_cache` | Alpha Vantage headline responses | 6h |
| `price_cache` | Latest quote for P/E computation | 15min |
| `audit_log` | Every MCP tool call (success + error) | none |

All tables are created on first import via `Base.metadata.create_all(engine)`, guarded against the race condition that occurs when multiple agent processes start concurrently (see [gotchas](#known-limitations)).

## Known limitations

- Citation grounding proves a cited number is real — it doesn't verify that prose *reasoning* about that number is correct (e.g. a claimed trend direction).
- Agent-to-agent calls use MCP, not the [A2A protocol](https://a2a-protocol.org/) — MCP is agent-to-tool by design; A2A would be the right choice if exposing these agents to external orchestrators becomes a goal.
- The 8-ticker golden set is sized to Alpha Vantage's 25-request/day free-tier quota.

## Running tests

```bash
pytest
```
