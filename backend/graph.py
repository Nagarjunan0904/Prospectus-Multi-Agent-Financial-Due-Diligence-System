"""
LangGraph due-diligence graph — wires the five agent nodes together and
attaches a PostgreSQL checkpointer for resumability.

Graph topology
--------------

    supervisor ──► data_agent ──┬──► quant_agent ──► risk_agent ──► END
                                └──► sentiment_agent ──────────────► END

data_agent fans out to quant_agent and sentiment_agent in parallel.
risk_agent runs after quant_agent completes (needs ratio_history).
sentiment_agent converges at END independently.

# TODO Phase 4: replace END with memo_writer node + conditional retry edges
# on both risk_agent and sentiment_agent once the memo_writer is implemented.

Checkpointing
-------------
PostgresSaver stores a checkpoint after each node completes.  Interrupted
runs (e.g. MCP server timeout) resume from the last successful node on the
next call with the same thread_id.  A completed run's final state is also
checkpointed, so calling ainvoke() again with the same thread_id returns
the cached final state without re-executing any nodes.

Usage
-----
    async with make_graph() as graph:
        state = await graph.ainvoke(
            {"ticker": "NVDA"},
            config={"configurable": {"thread_id": "run-nvda-001"}},
        )
"""
from __future__ import annotations

import contextlib
import os
from typing import AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from backend.agents import (
    data_agent,
    quant_agent,
    risk_agent,
    sentiment_agent,
    supervisor,
)
from backend.state import DueDiligenceState


# ---------------------------------------------------------------------------
# Graph builder (configured at import time; compiled at runtime with saver)
# ---------------------------------------------------------------------------

_builder = StateGraph(DueDiligenceState)

_builder.add_node("supervisor",       supervisor.run)
_builder.add_node("data_agent",       data_agent.run)
_builder.add_node("quant_agent",      quant_agent.run)
_builder.add_node("sentiment_agent",  sentiment_agent.run)
_builder.add_node("risk_agent",       risk_agent.run)

# Sequential spine
_builder.add_edge(START,          "supervisor")
_builder.add_edge("supervisor",   "data_agent")

# Fan-out: data_agent → quant_agent and sentiment_agent in parallel
_builder.add_edge("data_agent", "quant_agent")
_builder.add_edge("data_agent", "sentiment_agent")

# quant_agent feeds risk_agent; sentiment_agent is a leaf for now
_builder.add_edge("quant_agent",     "risk_agent")
_builder.add_edge("sentiment_agent", END)   # TODO Phase 4: replace END with memo_writer
_builder.add_edge("risk_agent",      END)   # TODO Phase 4: replace END with memo_writer


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def make_graph(
    interrupt_after: list[str] | None = None,
) -> AsyncGenerator:
    """Async context manager that yields a compiled, checkpointed graph.

    Opens a psycopg3 connection pool to Postgres (DATABASE_URL env var),
    creates the LangGraph checkpoint tables if they don't exist, compiles
    the graph with the checkpointer, and tears down the pool on exit.

    Parameters
    ----------
    interrupt_after:
        Node names after which execution should pause and the current state
        should be checkpointed.  Pass ``ainvoke(None, config)`` with the same
        ``thread_id`` to resume.  Omit (or pass ``None``) for normal
        uninterrupted execution.

    Example — normal run
    --------------------
    .. code-block:: python

        async with make_graph() as graph:
            state = await graph.ainvoke(
                {"ticker": "NVDA"},
                config={"configurable": {"thread_id": "run-001"}},
            )

    Example — interrupted / resumed run
    ------------------------------------
    .. code-block:: python

        cfg = {"configurable": {"thread_id": "run-001"}}
        async with make_graph(interrupt_after=["data_agent"]) as g:
            partial = await g.ainvoke({"ticker": "NVDA"}, config=cfg)

        async with make_graph() as g:          # no interrupt this time
            full = await g.ainvoke(None, config=cfg)   # None = resume
    """
    db_url = os.environ["DATABASE_URL"]
    async with AsyncPostgresSaver.from_conn_string(db_url) as checkpointer:
        await checkpointer.setup()   # idempotent; creates tables on first run
        compile_kwargs: dict = {"checkpointer": checkpointer}
        if interrupt_after:
            compile_kwargs["interrupt_after"] = interrupt_after
        yield _builder.compile(**compile_kwargs)
