"""FastAPI application for InvoiceTriageEnv."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from openenv.core.env_server import create_app

from invoice_triage_env.models import InvoiceAction, InvoiceObservation
from invoice_triage_env.server.invoice_triage_environment import (
    InvoiceTriageEnvironment,
)

app = create_app(
    env=InvoiceTriageEnvironment,
    action_cls=InvoiceAction,
    observation_cls=InvoiceObservation,
    env_name="InvoiceTriageEnv",
)

# ---------------------------------------------------------------------------
# Root route — serves dashboard so HF Spaces shows a UI when
# ENABLE_WEB_INTERFACE is NOT set (local dev).
# When ENABLE_WEB_INTERFACE=true, the Gradio app at /web handles the UI
# and HF's base_path: /web sends root traffic there.
# ---------------------------------------------------------------------------

DASHBOARD_HTML = Path(__file__).parent / ".." / "dashboard" / "index.html"


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the custom performance dashboard."""
    if DASHBOARD_HTML.exists():
        return HTMLResponse(content=DASHBOARD_HTML.read_text(), status_code=200)
    return HTMLResponse(
        content="<html><body><h1>Dashboard not found</h1></body></html>",
        status_code=404,
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
