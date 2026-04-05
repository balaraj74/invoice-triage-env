"""Serve the dashboard and provide a REST endpoint for live evaluation.

Usage:
    PYTHONPATH=. python -m invoice_triage_env.dashboard.serve
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from invoice_triage_env.examples.run_agent import run_agent_episode
from invoice_triage_env.tasks import ALL_TASKS

try:
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    import urllib.parse
except ImportError:
    raise SystemExit("Python stdlib http.server required")


DASHBOARD_DIR = Path(__file__).parent
RESULTS_CACHE: dict[str, Any] = {}


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve dashboard files and evaluation API."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/evaluate":
            self._handle_evaluate()
        elif parsed.path == "/api/tasks":
            self._json_response({"tasks": list(ALL_TASKS.keys())})
        else:
            super().do_GET()

    def _handle_evaluate(self) -> None:
        """Run all tasks and return results."""
        results = {}
        for tid in ALL_TASKS:
            task_def = ALL_TASKS[tid]
            t0 = time.time()
            episode = run_agent_episode(tid)
            elapsed = time.time() - t0
            results[tid] = {
                "reward": round(episode["reward"], 3),
                "steps": episode["steps"],
                "correct": episode["reward"] > 0,
                "subtasksDone": [
                    k for k, v in episode["progress"].items() if v
                ],
                "elapsed": round(elapsed, 3),
                "difficulty": task_def.difficulty,
            }
        self._json_response(results)

    def _json_response(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        """Quieter logging."""
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main() -> None:
    port = int(os.environ.get("DASHBOARD_PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), DashboardHandler)
    print(f"\n  🧾 InvoiceTriageEnv Dashboard")
    print(f"  ─────────────────────────────")
    print(f"  ➜ Dashboard:   http://localhost:{port}")
    print(f"  ➜ Evaluate:    http://localhost:{port}/api/evaluate")
    print(f"  ➜ Task list:   http://localhost:{port}/api/tasks\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
