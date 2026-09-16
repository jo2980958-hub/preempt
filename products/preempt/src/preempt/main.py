"""ASGI entry point: `uvicorn preempt.main:app`."""

from __future__ import annotations

from .service import create

app = create()
