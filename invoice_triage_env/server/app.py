"""FastAPI application for InvoiceTriageEnv."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from openenv.core.env_server import create_fastapi_app

from invoice_triage_env.models import InvoiceAction, InvoiceObservation
from invoice_triage_env.server.invoice_triage_environment import (
    InvoiceTriageEnvironment,
)

app = create_fastapi_app(
    env=InvoiceTriageEnvironment,
    action_cls=InvoiceAction,
    observation_cls=InvoiceObservation,
)

# ---------------------------------------------------------------------------
# Root route — serves dashboard so HF Spaces shows a UI
# ---------------------------------------------------------------------------

DASHBOARD_HTML = Path(__file__).parent / ".." / "dashboard" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the dashboard UI at the root path."""
    if DASHBOARD_HTML.exists():
        return HTMLResponse(content=DASHBOARD_HTML.read_text(), status_code=200)
    return HTMLResponse(
        content="""
        <html><body style="background:#0a0e1a;color:#f1f5f9;font-family:sans-serif;
        display:flex;align-items:center;justify-content:center;height:100vh;flex-direction:column">
        <h1>🧾 InvoiceTriageEnv</h1>
        <p>OpenEnv environment for AI agent invoice processing</p>
        <p style="color:#94a3b8;margin-top:20px">
        API endpoints: <code>/reset</code> <code>/step</code> <code>/state</code> <code>/docs</code>
        </p>
        </body></html>
        """,
        status_code=200,
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "environment": "InvoiceTriageEnv"})


def main() -> None:
    """Run the InvoiceTriageEnv server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
