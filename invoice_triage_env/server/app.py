"""FastAPI application for InvoiceTriageEnv."""

import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from openenv.core.env_server import create_app

from invoice_triage_env.models import InvoiceAction, InvoiceObservation
from invoice_triage_env.server.invoice_triage_environment import (
    InvoiceTriageEnvironment,
)
from invoice_triage_env.tasks import ALL_TASKS

app = create_app(
    env=InvoiceTriageEnvironment,
    action_cls=InvoiceAction,
    observation_cls=InvoiceObservation,
    env_name="InvoiceTriageEnv",
)

# ---------------------------------------------------------------------------
# Static assets
# ---------------------------------------------------------------------------

_OUTPUTS_DIR = Path(__file__).parent.parent.parent / "outputs"
_DASHBOARD_HTML = Path(__file__).parent / ".." / "dashboard" / "index.html"
_BENCHMARK_JSON = _OUTPUTS_DIR / "benchmark_results.json"

# Serve outputs/ (training curves, benchmark JSON) as static files
if _OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=str(_OUTPUTS_DIR)), name="outputs")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

from fastapi.responses import RedirectResponse

@app.get("/", response_class=RedirectResponse)
async def root_redirect():
    """Redirect root to the dashboard."""
    return RedirectResponse(url="/dashboard", status_code=302)

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the custom performance dashboard."""
    if _DASHBOARD_HTML.exists():
        return HTMLResponse(content=_DASHBOARD_HTML.read_text(), status_code=200)
    return HTMLResponse(
        content="<html><body><h1>Dashboard not found</h1></body></html>",
        status_code=404,
    )


# ---------------------------------------------------------------------------
# Benchmark API
# ---------------------------------------------------------------------------

@app.get("/api/benchmark")
async def get_benchmark():
    """Return the latest benchmark results from outputs/benchmark_results.json."""
    if _BENCHMARK_JSON.exists():
        data = json.loads(_BENCHMARK_JSON.read_text())
        return JSONResponse(data)
    return JSONResponse({"error": "No benchmark results yet. Run training first."}, status_code=404)


@app.post("/api/evaluate")
async def run_evaluate():
    """Run a quick 1-trial stochastic evaluation across all tasks and return results.

    Uses the latest checkpoint if available, otherwise a fresh random policy.
    This is intentionally lightweight — for full training, use train_reinforce.py.
    """
    import torch
    import random as _random

    from invoice_triage_env.training.obs_encoder import ObservationEncoder
    from invoice_triage_env.training.policy import ActorCriticPolicy
    from invoice_triage_env.training.train_reinforce import run_episode

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    encoder = ObservationEncoder()
    policy = ActorCriticPolicy().to(device)

    # Load best checkpoint if available
    ckpt_dir = _OUTPUTS_DIR / "checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("policy_ep*.pt"))
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location=device)
            policy.load_state_dict(ckpt["model_state"])

    policy.eval()
    results = {}
    with torch.no_grad():
        for tid, task in ALL_TASKS.items():
            env = InvoiceTriageEnvironment(task_id=tid)
            traj = run_episode(env, policy, encoder, tid, deterministic=False, seed=0)
            results[tid] = {
                "score": traj["final_score"],
                "steps": traj["steps"],
            }

    avg = sum(v["score"] for v in results.values()) / max(len(results), 1)
    return JSONResponse({
        "benchmark": results,
        "avg_benchmark_score": avg,
        "n_episodes": "live",
    })


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "environment": "InvoiceTriageEnv"})


def main() -> None:
    """Run the InvoiceTriageEnv server."""
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
