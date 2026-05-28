from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sef_gram.world_model import WorldBatch, pad_obs


@dataclass
class MLPWorldModelConfig:
    max_obs_dim: int = 16
    hidden_dim: int = 128
    num_actions: int = 4
    obs_weight: float = 1.0
    reward_weight: float = 1.0
    done_weight: float = 0.2


class MLPWorldModel(nn.Module):
    def __init__(self, cfg: MLPWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.max_obs_dim + cfg.num_actions
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
        )
        self.next_obs_head = nn.Linear(cfg.hidden_dim, cfg.max_obs_dim)
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)
        self.done_head = nn.Linear(cfg.hidden_dim, 1)

    def _features(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = pad_obs(obs, self.cfg.max_obs_dim)
        act = F.one_hot(actions.long().view(-1), num_classes=self.cfg.num_actions).float()
        return torch.cat([obs, act], dim=-1)

    def forward(self, batch: WorldBatch) -> Dict[str, torch.Tensor]:
        h = self.trunk(self._features(batch.obs, batch.actions))
        return {
            "pred_next_obs": self.next_obs_head(h),
            "target_next_obs": pad_obs(batch.next_obs, self.cfg.max_obs_dim),
            "pred_reward": self.reward_head(h).squeeze(-1),
            "target_reward": batch.rewards.float().view(-1),
            "pred_done_logit": self.done_head(h).squeeze(-1),
            "target_done": batch.dones.float().view(-1),
        }

    def loss(self, batch: WorldBatch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(batch)
        obs_mse = F.mse_loss(out["pred_next_obs"], out["target_next_obs"])
        reward_mse = F.mse_loss(out["pred_reward"], out["target_reward"])
        done_bce = F.binary_cross_entropy_with_logits(out["pred_done_logit"], out["target_done"])
        total = self.cfg.obs_weight * obs_mse + self.cfg.reward_weight * reward_mse + self.cfg.done_weight * done_bce
        return total, {
            "total": total.detach(),
            "obs_mse": obs_mse.detach(),
            "reward_mse": reward_mse.detach(),
            "done_bce": done_bce.detach(),
        }

    @torch.no_grad()
    def predict_step(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        dummy = WorldBatch(
            obs=obs,
            actions=actions,
            next_obs=torch.zeros(obs.shape[0], self.cfg.max_obs_dim, device=obs.device),
            rewards=torch.zeros(obs.shape[0], device=obs.device),
            dones=torch.zeros(obs.shape[0], device=obs.device),
        )
        out = self.forward(dummy)
        return {
            "next_obs": out["pred_next_obs"],
            "reward": out["pred_reward"],
            "done_prob": torch.sigmoid(out["pred_done_logit"]),
        }


@dataclass
class GRUWorldModelConfig:
    max_obs_dim: int = 16
    hidden_dim: int = 128
    num_actions: int = 4
    obs_weight: float = 1.0
    reward_weight: float = 1.0
    done_weight: float = 0.2


class GRUWorldModel(nn.Module):
    """Recurrent baseline for partially observable world-model tasks.

    This is a stronger memory baseline than MLP. It keeps a GRU hidden state across
    rollout steps and predicts next observation, reward, and done from the hidden
    state after consuming [obs_t, action_t].
    """

    def __init__(self, cfg: GRUWorldModelConfig):
        super().__init__()
        self.cfg = cfg
        input_dim = cfg.max_obs_dim + cfg.num_actions
        self.input_proj = nn.Sequential(nn.Linear(input_dim, cfg.hidden_dim), nn.SiLU())
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.next_obs_head = nn.Linear(cfg.hidden_dim, cfg.max_obs_dim)
        self.reward_head = nn.Linear(cfg.hidden_dim, 1)
        self.done_head = nn.Linear(cfg.hidden_dim, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.cfg.hidden_dim, device=device)

    def _features(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = pad_obs(obs, self.cfg.max_obs_dim)
        act = F.one_hot(actions.long().view(-1), num_classes=self.cfg.num_actions).float()
        return self.input_proj(torch.cat([obs, act], dim=-1))

    def step(self, obs: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden = self.gru(self._features(obs, actions), hidden)
        out = {
            "pred_next_obs": self.next_obs_head(hidden),
            "pred_reward": self.reward_head(hidden).squeeze(-1),
            "pred_done_logit": self.done_head(hidden).squeeze(-1),
        }
        return hidden, out

    def forward(self, batch: WorldBatch) -> Dict[str, torch.Tensor]:
        hidden = self.initial_state(batch.obs.shape[0], batch.obs.device)
        _, pred = self.step(batch.obs, batch.actions, hidden)
        return {
            "pred_next_obs": pred["pred_next_obs"],
            "target_next_obs": pad_obs(batch.next_obs, self.cfg.max_obs_dim),
            "pred_reward": pred["pred_reward"],
            "target_reward": batch.rewards.float().view(-1),
            "pred_done_logit": pred["pred_done_logit"],
            "target_done": batch.dones.float().view(-1),
        }

    def loss(self, batch: WorldBatch) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(batch)
        obs_mse = F.mse_loss(out["pred_next_obs"], out["target_next_obs"])
        reward_mse = F.mse_loss(out["pred_reward"], out["target_reward"])
        done_bce = F.binary_cross_entropy_with_logits(out["pred_done_logit"], out["target_done"])
        total = self.cfg.obs_weight * obs_mse + self.cfg.reward_weight * reward_mse + self.cfg.done_weight * done_bce
        return total, {
            "total": total.detach(),
            "obs_mse": obs_mse.detach(),
            "reward_mse": reward_mse.detach(),
            "done_bce": done_bce.detach(),
        }

    @torch.no_grad()
    def predict_step(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        hidden = self.initial_state(obs.shape[0], obs.device)
        _, pred = self.step(obs, actions, hidden)
        return {
            "next_obs": pred["pred_next_obs"],
            "reward": pred["pred_reward"],
            "done_prob": torch.sigmoid(pred["pred_done_logit"]),
        }
