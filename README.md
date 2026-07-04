# Prospectus — Multi-Agent Financial Due-Diligence System

A LangGraph orchestrator driving four specialised MCP servers against
SEC EDGAR, FinBERT sentiment, quant signals, and a risk synthesiser.

## Quick start

```bash
# 1. Start Postgres
docker-compose up -d

# 2. Install deps
pip install -r requirements.txt

# 3. Copy and fill env vars
cp .env.example .env

# 4. Run the data-agent MCP server (HTTP, port 9001)
python -m mcp_servers.data_agent.server

# 5. Or run it over stdio for Claude Desktop testing
python -m mcp_servers.data_agent.server --transport stdio
```

## Architecture

```
Orchestrator (LangGraph / GPT-4.1-mini)
    │
    ├── data_agent      :9001  SEC EDGAR — filings, facts, insider trades
    ├── quant_agent     :9002  price signals, volatility, factor models
    ├── sentiment_agent :9003  FinBERT on 10-K / news headlines
    └── risk_agent      :9004  synthesises all three into a risk score
```

Each agent is a standalone `mcp.server.Server` served over
streamable-HTTP, talking to a shared Postgres instance for caching and
audit logging.

## Authentication — design decision (important for interviews)

### What's here

The data-agent server uses a **static bearer token** (`MCP_DATA_AGENT_TOKEN`)
checked by `_BearerAuthMiddleware` on every HTTP request.  Set it in `.env`:

```
MCP_DATA_AGENT_TOKEN=some-long-random-string
```

Clients send:

```
Authorization: Bearer some-long-random-string
```

For **stdio transport** (Claude Desktop local testing) there is no auth
check — stdio is a local-process pipe, so network-level auth is moot.

### What production replaces this with

The MCP specification (§4.3) mandates **OAuth 2.1** for production
deployments:

| Requirement | RFC / spec |
|---|---|
| Discovery endpoint | RFC 8414 `/.well-known/oauth-authorization-server` |
| PKCE-protected authorisation code flow | RFC 7636 |
| Short-lived, scoped access tokens | e.g. `read:edgar`, `read:insider` |
| Token introspection on every request | RFC 7662 |

The `_BearerAuthMiddleware` class is the **exact injection point** for
this swap — replace the static string comparison with a JWKS-verified JWT
check or a token-introspection call and nothing else in the server changes.

**This is a deliberate portfolio-scope decision, not an oversight.**
Adding full OIDC plumbing would add ~300 lines of boilerplate that
obscures the agent architecture being demonstrated.  In a production
engagement this would be handled by an API gateway (Kong, AWS API GW,
etc.) in front of the MCP servers so the server code stays identical.

## Database tables

| Table | Purpose | TTL |
|---|---|---|
| `ticker_cache` | SEC ticker→CIK map | 24 h |
| `filing_cache` | submissions + company-facts blobs | 24 h |
| `document_cache` | raw 10-K/10-Q HTML | none (immutable) |
| `form4_cache` | Form 4 XML | none (immutable) |
| `audit_log` | every tool call (success + error) | none |

All tables are created automatically on first import via
`Base.metadata.create_all(engine)`.

## Running tests

```bash
pytest
```
