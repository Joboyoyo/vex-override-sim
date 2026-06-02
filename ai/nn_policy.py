"""Tiny neural-network policy for the toggle duel.

A 2-layer MLP that maps observation → action logits. Designed to be plugged
into the existing duel mode as a drop-in replacement for ScriptedBot's
decisions: given the env's observation vector, output one of the 9 discrete
actions defined in envs.toggle_duel_env.DISCRETE_ACTIONS.

The same class is used during training (sampling actions stochastically) and
inference (greedy argmax) — controlled by the `greedy=` flag on `act()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """Two hidden layers, tanh activations. Outputs categorical logits over
    the 9 discrete actions and a scalar value baseline for advantage RL."""

    def __init__(self, obs_dim: int = 14, action_dim: int = 9,
                 hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, action_dim)
        self.value_head  = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, obs, greedy: bool = False) -> int:
        """Sample (or argmax) a discrete action for a single observation.
        Accepts numpy arrays or torch tensors."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.forward(obs)
            if greedy:
                return int(logits.argmax(dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    # --- persistence ---

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), str(path))

    @classmethod
    def load(cls, path: str | Path, obs_dim: Optional[int] = None,
             action_dim: Optional[int] = None,
             hidden: Optional[int] = None) -> "MLPPolicy":
        """Load a saved policy. Auto-detects layer sizes from the state_dict
        so checkpoints trained with different hidden widths just work."""
        sd = torch.load(str(path), map_location="cpu")
        # shared.0.weight has shape [hidden, obs_dim]
        if "shared.0.weight" in sd:
            inferred_hidden, inferred_obs = sd["shared.0.weight"].shape
        else:
            inferred_hidden = inferred_obs = None
        # policy_head.weight has shape [action_dim, hidden]
        if "policy_head.weight" in sd:
            inferred_actions = sd["policy_head.weight"].shape[0]
        else:
            inferred_actions = None
        net = cls(
            obs_dim=obs_dim if obs_dim is not None else inferred_obs or 14,
            action_dim=action_dim if action_dim is not None else inferred_actions or 9,
            hidden=hidden if hidden is not None else inferred_hidden or 64,
        )
        net.load_state_dict(sd)
        net.eval()
        return net


__all__ = ["MLPPolicy"]
