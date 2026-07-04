"""
Audit log: every MCP tool invocation writes one row — including failures.

Table (auto-created on first import, idempotent):
  audit_log — (id, agent, tool, params_hash, started_at, latency_ms, status)

params_hash = sha256 of the sorted, JSON-serialised tool kwargs so identical
calls can be correlated without storing potentially-sensitive arguments.
"""
import hashlib
import json
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from shared.db import Base, SessionLocal, engine


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    agent = Column(String(64), nullable=False)
    tool = Column(String(128), nullable=False)
    params_hash = Column(String(64), nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=False)
    latency_ms = Column(Float, nullable=True)
    status = Column(String(16), nullable=True)  # 'success' | 'error'


Base.metadata.create_all(bind=engine)


def make_params_hash(kwargs: dict) -> str:
    """sha256 of sorted, JSON-serialised tool kwargs."""
    return hashlib.sha256(
        json.dumps(kwargs, sort_keys=True, default=str).encode()
    ).hexdigest()


def record(
    agent: str,
    tool: str,
    params_hash: str,
    started_at: datetime,
    latency_ms: float,
    status: str,
) -> None:
    """Write one audit row synchronously.

    Call via ``asyncio.to_thread(record, ...)`` from async contexts so the
    synchronous DB write doesn't block the event loop.
    """
    with SessionLocal() as session:
        session.add(
            AuditLog(
                agent=agent,
                tool=tool,
                params_hash=params_hash,
                started_at=started_at,
                latency_ms=latency_ms,
                status=status,
            )
        )
        session.commit()
