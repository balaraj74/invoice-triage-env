"""Server entry point for InvoiceTriageEnv.

This module provides the ASGI app and CLI entry point for the
InvoiceTriageEnv OpenEnv server.
"""

from invoice_triage_env.server.app import app, main  # noqa: F401


def main() -> None:
    """Run the InvoiceTriageEnv server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
