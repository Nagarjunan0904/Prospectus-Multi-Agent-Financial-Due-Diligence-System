"""
Platform-specific fixes that must be applied before the first asyncio.run().
Import and call apply_windows_event_loop_fix() at the top of every entrypoint
that touches the LangGraph graph or its AsyncPostgresSaver checkpointer.
"""
import asyncio
import sys


def apply_windows_event_loop_fix() -> None:
    # psycopg3's AsyncPostgresSaver requires SelectorEventLoop; Windows
    # defaults to ProactorEventLoop (Python 3.8+), which is incompatible.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
