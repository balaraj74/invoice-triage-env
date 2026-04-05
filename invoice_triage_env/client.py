"""EnvClient wrapper for InvoiceTriageEnv.

Provides a typed client that connects to the HTTP server and
deserializes responses into InvoiceObservation / InvoiceState objects.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core.env_client import EnvClient

from invoice_triage_env.models import InvoiceAction, InvoiceObservation, InvoiceState


class InvoiceTriageClient(EnvClient[InvoiceAction, InvoiceObservation, InvoiceState]):
    """Typed HTTP client for the InvoiceTriageEnv server."""

    def __init__(self, server_url: str = "http://localhost:8000") -> None:
        super().__init__(
            server_url=server_url,
            action_cls=InvoiceAction,
            observation_cls=InvoiceObservation,
            state_cls=InvoiceState,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvoiceObservation:
        """Reset and get the initial observation."""
        return super().reset(seed=seed, episode_id=episode_id, **kwargs)

    def step(self, action: InvoiceAction, **kwargs: Any) -> InvoiceObservation:
        """Send an action and receive an observation."""
        return super().step(action=action, **kwargs)

    @property
    def state(self) -> InvoiceState:
        """Get current environment state."""
        return super().state
