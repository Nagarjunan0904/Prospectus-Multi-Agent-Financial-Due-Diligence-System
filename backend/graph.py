"""
LangGraph due-diligence graph — wires all seven agent nodes together and
attaches a PostgreSQL checkpointer for resumability.

Graph topology
--------------

    supervisor ──► data_agent ──┬──► quant_agent ──► risk_agent ──┐
                                └──► sentiment_agent ──────────────┤
                                                                   ▼
                                                            memo_writer
                                                                   │
                                                            self_critic
                                                           ╱           ╲
                                                    (retry=None)  (retry=node)
                                                         │               │
                                                        END         that node ──► ...

data_agent fans out to quant_agent and sentiment_agent in parallel.
risk_agent runs after quant_agent (needs ratio_history).
Both risk_agent and sentiment_agent converge at memo_writer.
self_critic either ends the run (retry_agent is None) or routes back to
the agent that produced the most ungrounded claims (or memo_writer itself
when completeness is the only issue).

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
    memo  = state.get("memo")       # InvestmentMemo.model_dump() or None
    retry = state.get("retry_agent")  # None when self_critic approved the memo
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import AsyncGenerator

_log = logging.getLogger(__name__)

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from backend.agents import (
    data_agent,
    memo_writer,
    quant_agent,
    risk_agent,
    self_critic,
    sentiment_agent,
    supervisor,
)
from backend.state import DueDiligenceState


# ---------------------------------------------------------------------------
# Graph builder (configured at import time; compiled at runtime with saver)
# ---------------------------------------------------------------------------

MAX_RETRIES = 2  # blueprint spec 7.4: cap the self_critic → upstream retry loop

_RETRYABLE_NODES = frozenset(
    {"quant_agent", "sentiment_agent", "risk_agent", "data_agent", "memo_writer"}
)


def _route_self_critic(state: DueDiligenceState) -> str:
    """Conditional edge out of self_critic.

    Returns the name of the node to run next:
      • END          -- memo is acceptable (retry_agent is None), OR
                        the retry cap (MAX_RETRIES) has been reached
      • <node_name>  -- re-run that upstream node (then flow continues
                        through the graph back to self_critic)

    Unknown node names and cap-exceeded cases both fall back to END.
    """
    retry = state.get("retry_agent")
    if retry and retry in _RETRYABLE_NODES:
        count = state.get("retry_count", 0)
        if count < MAX_RETRIES:
            return retry
        _log.warning(
            "self_critic: retry cap reached (%d/%d) for retry_agent=%r — "
            "accepting memo as low-confidence result",
            count, MAX_RETRIES, retry,
        )
    return END


_builder = StateGraph(DueDiligenceState)

_builder.add_node("supervisor",       supervisor.run)
_builder.add_node("data_agent",       data_agent.run)
_builder.add_node("quant_agent",      quant_agent.run)
_builder.add_node("sentiment_agent",  sentiment_agent.run)
_builder.add_node("risk_agent",       risk_agent.run)
# defer=True: memo_writer has two predecessors that arrive in different supersteps
# (sentiment_agent at depth N+2, risk_agent at depth N+3 via quant_agent).
# Without defer, LangGraph fires memo_writer once per predecessor as each
# superstep settles, causing two LLM calls per run.  defer=True holds the node
# until ALL predecessors in the current execution have completed.
_builder.add_node("memo_writer",      memo_writer.run, defer=True)
_builder.add_node("self_critic",      self_critic.run)

# Sequential spine
_builder.add_edge(START,          "supervisor")
_builder.add_edge("supervisor",   "data_agent")

# Fan-out: data_agent → quant_agent and sentiment_agent in parallel
_builder.add_edge("data_agent",       "quant_agent")
_builder.add_edge("data_agent",       "sentiment_agent")

# quant_agent feeds risk_agent (needs ratio_history)
_builder.add_edge("quant_agent",     "risk_agent")

# Both parallel branches converge at memo_writer
_builder.add_edge("sentiment_agent", "memo_writer")
_builder.add_edge("risk_agent",      "memo_writer")

# memo_writer feeds self_critic
_builder.add_edge("memo_writer",     "self_critic")

# Conditional exit: self_critic either ends or retries an upstream node
_builder.add_conditional_edges("self_critic", _route_self_critic)


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
