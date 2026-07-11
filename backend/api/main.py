"""
FastAPI application for the due-diligence pipeline.

Startup
-------
The LangGraph compiled graph (with its PostgreSQL checkpointer pool) is
created ONCE in the lifespan context manager and stored on app.state.graph.
Every endpoint reads it from there — there is no per-request graph setup.

MCP clients deliberately open a fresh session per call (see data_client.py),
so there is nothing to initialise here for them at startup.

Endpoints
---------
GET  /health               — DB + MCP server liveness probes
POST /diligence            — synchronous full run; blocks until memo is ready
GET  /diligence/stream     — SSE stream of real-time node updates
GET  /memo/{run_id}        — retrieve final state for a completed run
GET  /eval                 — evaluation metrics (Phase 6 stub)

Run
---
    uvicorn backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from backend.graph import make_graph  # noqa: E402 — must follow load_dotenv()

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — graph opened once at startup, shared across all requests
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Open the LangGraph graph (and its psycopg3 pool) at startup; close on shutdown."""
    async with make_graph() as graph:
        app.state.graph = graph
        _log.info("Graph ready; checkpointer pool open")
        yield
    _log.info("Graph shut down; checkpointer pool closed")


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Due-Diligence API", version="1.0", lifespan=lifespan)

# CORS_ORIGINS is a comma-separated list of allowed origins.
# Default covers Vite's dev server; override in .env for production.
_CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class DiligenceRequest(BaseModel):
    ticker: str


class DiligenceResponse(BaseModel):
    run_id: str
    memo: dict[str, Any] | None
    ratios: dict[str, Any] | None
    sentiment: dict[str, Any] | None
    risk_flags: list[dict[str, Any]] | None
    citation_coverage: float
    attempts: int
    errors: list[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _check_db() -> str:
    """Return 'up' if DATABASE_URL is reachable, 'down' otherwise."""
    try:
        db_url = os.environ["DATABASE_URL"]
        async with await psycopg.AsyncConnection.connect(db_url) as conn:
            await conn.execute("SELECT 1")
        return "up"
    except Exception as exc:
        _log.warning("health/db: %s", exc)
        return "down"


async def _check_mcp(name: str, url: str, token: str, timeout: float = 3.0) -> str:
    """Probe an MCP server with a list_tools call; the cheapest valid RPC."""
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        async with asyncio.timeout(timeout):
            async with streamablehttp_client(url, headers=headers) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    await session.list_tools()
        return "up"
    except Exception as exc:
        _log.warning("health/mcp/%s: %s", name, exc)
        return "down"


def _sse(event: dict[str, Any]) -> str:
    """Format a dict as a Server-Sent Events data line."""
    return f"data: {json.dumps(event, default=str)}\n\n"


def _build_response(run_id: str, state: dict[str, Any]) -> DiligenceResponse:
    return DiligenceResponse(
        run_id=run_id,
        memo=state.get("memo"),
        ratios=state.get("ratios"),
        sentiment=state.get("sentiment"),
        risk_flags=state.get("risk_flags"),
        citation_coverage=state.get("citation_coverage", 0.0),
        attempts=(state.get("retry_count") or 0) + 1,
        errors=state.get("errors") or [],
    )


# Fields emitted as state_update events during streaming.
# agent_trace is handled separately (one event per entry, not as a bulk key).
_STREAM_KEYS = frozenset({
    "cik", "ratios", "ratio_history", "sentiment",
    "risk_flags", "memo", "citation_coverage",
})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Liveness check: PostgreSQL + all four MCP servers.

    Response shape::

        {
          "status": "ok" | "degraded",
          "db": "up" | "down",
          "mcp_servers": {"data": "up"|"down", "quant": ..., "sentiment": ..., "risk": ...}
        }
    """
    db_status, *mcp_results = await asyncio.gather(
        _check_db(),
        _check_mcp("data",
                   os.environ.get("MCP_DATA_AGENT_URL",       "http://localhost:9001/mcp"),
                   os.environ.get("MCP_DATA_AGENT_TOKEN",      "")),
        _check_mcp("quant",
                   os.environ.get("MCP_QUANT_AGENT_URL",      "http://localhost:9002/mcp"),
                   os.environ.get("MCP_QUANT_AGENT_TOKEN",     "")),
        _check_mcp("sentiment",
                   os.environ.get("MCP_SENTIMENT_AGENT_URL",  "http://localhost:9003/mcp"),
                   os.environ.get("MCP_SENTIMENT_AGENT_TOKEN", "")),
        _check_mcp("risk",
                   os.environ.get("MCP_RISK_AGENT_URL",       "http://localhost:9004/mcp"),
                   os.environ.get("MCP_RISK_AGENT_TOKEN",      "")),
    )
    mcp_status = dict(zip(("data", "quant", "sentiment", "risk"), mcp_results))
    overall = (
        "ok"
        if db_status == "up" and all(v == "up" for v in mcp_status.values())
        else "degraded"
    )
    return {"status": overall, "db": db_status, "mcp_servers": mcp_status}


@app.post("/diligence", response_model=DiligenceResponse)
async def run_diligence(req: DiligenceRequest) -> DiligenceResponse:
    """
    Run the full due-diligence pipeline synchronously.

    Generates a UUID ``run_id`` that doubles as the LangGraph ``thread_id``
    so the same ID can be used with ``GET /memo/{run_id}`` to re-fetch the
    result later from the checkpoint store.

    Node-level failures (e.g. a single MCP call timing out) are captured
    inside the graph and returned in the ``errors`` list — they do not raise
    here.  Only an exception that escapes the graph entirely becomes a 500.
    """
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}
    graph = app.state.graph  # type: ignore[attr-defined]
    try:
        state: dict[str, Any] = await graph.ainvoke(
            {"ticker": req.ticker.upper()},
            config=config,
        )
    except Exception as exc:
        _log.exception(
            "graph.ainvoke failed for ticker=%s run_id=%s", req.ticker, run_id
        )
        raise HTTPException(
            status_code=500,
            detail={"error": str(exc), "run_id": run_id},
        )
    return _build_response(run_id, state)


@app.get("/diligence/stream")
async def stream_diligence(
    ticker: str = Query(..., description="Ticker symbol, e.g. NVDA"),
) -> StreamingResponse:
    """
    Run the pipeline and stream node updates as Server-Sent Events.

    SSE event types
    ---------------
    ``run_id``      — first event; contains the run ID for later ``/memo/{run_id}``
    ``trace_entry`` — one per agent_trace entry; emitted as each node completes
    ``state_update``— emitted when a key in _STREAM_KEYS is updated by a node
    ``error``       — emitted if an exception escapes the graph
    ``end``         — always the last event (after success or error)

    Retry cycles (memo_writer → self_critic → upstream node → ...) emit
    fresh trace_entry events for each re-executed node — this is intentional
    so the retry can be watched live.
    """
    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}
    graph = app.state.graph  # type: ignore[attr-defined]

    async def event_stream() -> AsyncGenerator[str, None]:
        yield _sse({"type": "run_id", "run_id": run_id})
        try:
            async for chunk in graph.astream(
                {"ticker": ticker.upper()},
                config=config,
                stream_mode="updates",
            ):
                # chunk = {node_name: {key: value, ...}, ...}
                for node_name, node_update in chunk.items():
                    # One trace_entry event per agent_trace item in this update.
                    for entry in node_update.get("agent_trace") or []:
                        yield _sse({"type": "trace_entry", **entry})
                    # One state_update event per result key that changed.
                    for key in _STREAM_KEYS:
                        if key in node_update:
                            yield _sse({
                                "type":  "state_update",
                                "key":   key,
                                "value": node_update[key],
                            })
            yield _sse({"type": "end"})
        except Exception as exc:
            _log.exception(
                "stream failed for ticker=%s run_id=%s", ticker, run_id
            )
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "end"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",    # tells Nginx not to buffer this response
        },
    )


@app.get("/memo/{run_id}")
async def get_memo(run_id: str) -> dict[str, Any]:
    """
    Retrieve the final checkpointed state for a completed run.

    Uses LangGraph's ``aget_state`` to read the latest checkpoint for the
    given ``thread_id`` (= ``run_id``).  Does not query a separate results
    table — the checkpoint IS the source of truth.

    Returns 404 if no checkpoint exists (run_id was never started or the
    checkpoint store was cleared).
    """
    graph = app.state.graph  # type: ignore[attr-defined]
    config = {"configurable": {"thread_id": run_id}}
    try:
        snapshot = await graph.aget_state(config=config)
    except Exception as exc:
        _log.warning("aget_state failed for run_id=%s: %s", run_id, exc)
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint found for run_id={run_id!r}",
        )
    if not snapshot or not snapshot.values:
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint found for run_id={run_id!r}",
        )
    return _build_response(run_id, snapshot.values).model_dump()


_EVAL_RESULTS_PATH = Path(__file__).parent.parent.parent / "data" / "eval_results.json"

_EVAL_STUB: dict[str, Any] = {
    "run_timestamp":         None,
    "tickers_evaluated":     0,
    "ratio_accuracy":        None,
    "avg_citation_coverage": None,
    "retry_rate":            None,
    "red_flag_precision":    None,
    "red_flag_recall":       None,
    "latency_p50":           None,
    "latency_p95":           None,
    "per_ticker":            {},
}


@app.get("/eval")
async def eval_stats() -> dict[str, Any]:
    """
    Evaluation metrics from the last offline eval run.

    Returns the content of data/eval_results.json if it exists (written by
    backend/evaluation/eval_pipeline.py), otherwise returns a zero-valued
    stub with the same schema so callers never crash on missing fields.

    Run the eval manually:
        python backend/evaluation/eval_pipeline.py
        python backend/evaluation/eval_pipeline.py --tickers NVDA MSFT
    """
    if _EVAL_RESULTS_PATH.exists():
        return json.loads(_EVAL_RESULTS_PATH.read_text(encoding="utf-8"))
    return _EVAL_STUB
