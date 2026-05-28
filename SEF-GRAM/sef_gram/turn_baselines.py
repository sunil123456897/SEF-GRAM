from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


class MLPTurnBaseline(nn.Module):
    """Feedforward MLP baseline — no memory, re-encodes obs each turn."""

    def __init__(self, obs_dim: int, num_actions: int, env_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.env_vocab_size = env_vocab_size
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.env_head = nn.Linear(hidden_dim, env_vocab_size)

    def initial_state(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        flat = obs.reshape(-1, obs.shape[-1])
        h = self.trunk(flat)
        dummy = torch.zeros(flat.shape[0], 1, device=obs.device)
        return h, dummy, torch.zeros_like(h[:, :1])

    def forward_turn(
        self, z: Tensor, memory: Tensor, actions: Tensor, obs: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if obs is not None:
            flat = obs.reshape(-1, obs.shape[-1])
            h = self.trunk(flat)
            return h, memory
        return z, memory

    def policy_logits(self, z: Tensor) -> Tensor:
        return self.policy_head(z)

    def env_logits_from_z(self, z: Tensor) -> Optional[Tensor]:
        return self.env_head(z)


class GRUTurnBaseline(nn.Module):
    """GRU recurrent baseline — hidden state carries memory across turns."""

    def __init__(self, obs_dim: int, num_actions: int, env_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.env_vocab_size = env_vocab_size
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.env_head = nn.Linear(hidden_dim, env_vocab_size)

    def initial_state(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        flat = obs.reshape(-1, obs.shape[-1])
        x = self.obs_encoder(flat)
        h = self.gru(x, torch.zeros(flat.shape[0], self.hidden_dim, device=obs.device))
        dummy = torch.zeros(flat.shape[0], 1, device=obs.device)
        return h, dummy, torch.zeros_like(h[:, :1])

    def forward_turn(
        self, z: Tensor, memory: Tensor, actions: Tensor, obs: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if obs is not None:
            flat = obs.reshape(-1, obs.shape[-1])
            x = self.obs_encoder(flat)
            h = self.gru(x, z)
            return h, memory
        return z, memory

    def policy_logits(self, z: Tensor) -> Tensor:
        return self.policy_head(z)

    def env_logits_from_z(self, z: Tensor) -> Optional[Tensor]:
        return self.env_head(z)


class LSTMTurnBaseline(nn.Module):
    """LSTM recurrent baseline — separate cell state for stronger memory."""

    def __init__(self, obs_dim: int, num_actions: int, env_vocab_size: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.env_vocab_size = env_vocab_size
        self.hidden_dim = hidden_dim

        self.obs_encoder = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.env_head = nn.Linear(hidden_dim, env_vocab_size)

    def initial_state(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        flat = obs.reshape(-1, obs.shape[-1])
        x = self.obs_encoder(flat)
        h0 = torch.zeros(flat.shape[0], self.hidden_dim, device=obs.device)
        c0 = torch.zeros(flat.shape[0], self.hidden_dim, device=obs.device)
        h, c = self.lstm(x, (h0, c0))
        return h, c, torch.zeros_like(h[:, :1])

    def forward_turn(
        self, z: Tensor, memory: Tensor, actions: Tensor, obs: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if obs is not None:
            flat = obs.reshape(-1, obs.shape[-1])
            x = self.obs_encoder(flat)
            h, c = self.lstm(x, (z, memory))
            return h, c
        return z, memory

    def policy_logits(self, z: Tensor) -> Tensor:
        return self.policy_head(z)

    def env_logits_from_z(self, z: Tensor) -> Optional[Tensor]:
        return self.env_head(z)
