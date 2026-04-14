"""ActorCriticPolicy — PyTorch policy and value network for InvoiceTriageEnv.

Architecture: two-headed MLP with shared encoder.

    Input:  obs_tensor  (batch, OBS_DIM=316)
    ↓
    Shared encoder:  Linear(316→512) → LayerNorm → GELU
                     Linear(512→256) → LayerNorm → GELU
                     Residual dropout
    ↓
    Policy head:  Linear(256→N_ACTIONS) → Categorical distribution
    Value  head:  Linear(256→1)         → scalar V(s)

Design notes:
  - LayerNorm stabilises training on the sparse binary features.
  - GELU outperforms ReLU on structured tabular-style inputs.
  - The shared encoder ensures value and policy stay aligned.
  - Designed for REINFORCE and PPO (returns both logits + value).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from invoice_triage_env.training.obs_encoder import N_ACTIONS, OBS_DIM

HIDDEN_DIM = 256


class SharedEncoder(nn.Module):
    """Two-layer MLP with residual-like structure shared between heads."""

    def __init__(self, in_dim: int = OBS_DIM, hidden: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden * 2),
            nn.LayerNorm(hidden * 2),
            nn.GELU(),
            nn.Dropout(p=0.1),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        # Projection for skip connection
        self.skip_proj = nn.Linear(in_dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.skip_proj(x)
        h = self.layer1(x)
        h = self.layer2(h)
        return h + skip  # residual


class ActorCriticPolicy(nn.Module):
    """Combined Actor-Critic for REINFORCE / PPO on InvoiceTriageEnv.

    Usage::

        policy = ActorCriticPolicy()
        obs_tensor = encoder.encode(obs)          # shape: (OBS_DIM,)

        # Forward pass
        action_logits, value = policy(obs_tensor.unsqueeze(0))

        # Sample an action
        dist = policy.distribution(obs_tensor.unsqueeze(0))
        action_idx = dist.sample()
        log_prob   = dist.log_prob(action_idx)
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: int = HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.encoder = SharedEncoder(in_dim=obs_dim, hidden=hidden)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

        # Orthogonal initialisation — recommended for RL
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Policy head: smaller gain → less confident initial distribution
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: shape (batch, OBS_DIM)

        Returns:
            logits: shape (batch, N_ACTIONS)
            value:  shape (batch, 1)
        """
        h = self.encoder(obs)
        return self.policy_head(h), self.value_head(h)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        """Return a Categorical distribution over actions given obs."""
        logits, _ = self.forward(obs)
        return Categorical(logits=logits)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Sample or select best action.

        Returns:
            action_idx:  int
            log_prob:    scalar tensor
            value:       scalar tensor
        """
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(-1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
