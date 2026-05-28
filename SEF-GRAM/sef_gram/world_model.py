from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sef_gram.full_system import (
    SEFGRAMConfig,
    StochasticRecursiveWorldModel,
    diagonal_kl_to_standard_normal,
)


Tensor = torch.Tensor


@dataclass
class WorldBatch:
    """Common batch format for universal world-model training.

    All observations are already projected/padded into the shared observation
    interface. Actions are discrete for the first MVP; continuous controls should
    be quantized by the environment adapter.
    """

    obs: Tensor
    actions: Tensor
    next_obs: Tensor
    rewards: Tensor
    dones: Tensor
    env_ids: Optional[Tensor] = None

    def to(self, device: torch.device) -> "WorldBatch":
        return WorldBatch(
            obs=self.obs.to(device),
            actions=self.actions.to(device),
            next_obs=self.next_obs.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            env_ids=None if self.env_ids is None else self.env_ids.to(device),
        )


@dataclass
class WorldModelConfig:
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    beta_kl: float = 1e-3
    latent_weight: float = 1.0
    obs_weight: float = 1.0
    reward_weight: float = 1.0
    done_weight: float = 0.2
    logvar_weight: float = 1e-3


def pad_obs(obs: Tensor, max_obs_dim: int) -> Tensor:
    if obs.shape[-1] == max_obs_dim:
        return obs.float()
    if obs.shape[-1] > max_obs_dim:
        raise ValueError(f"obs dim {obs.shape[-1]} exceeds max_obs_dim={max_obs_dim}")
    pad_width = max_obs_dim - obs.shape[-1]
    return F.pad(obs.float(), (0, pad_width))


class UniversalWorldModel(nn.Module):
    """SEF-GRAM Phase 2 universal latent world-model core.

    This wraps the existing stochastic latent core in a stable world-model API:
    observation_t + action_t -> predicted latent/observation/reward/done at t+1.
    Different environments plug in through adapters/generators that emit WorldBatch.
    """

    def __init__(self, cfg: WorldModelConfig):
        super().__init__()
        self.cfg = cfg
        self.core = StochasticRecursiveWorldModel(
            SEFGRAMConfig(
                input_dim=cfg.max_obs_dim,
                latent_dim=cfg.latent_dim,
                hidden_dim=cfg.hidden_dim,
                num_actions=cfg.num_actions,
                recursion_depth=1,
                num_trajectories=1,
                reward_names=("reward",),
            )
        )
        self.obs_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.max_obs_dim),
        )
        self.reward_head = nn.Linear(cfg.latent_dim, 1)
        self.done_head = nn.Linear(cfg.latent_dim, 1)

    def forward(self, batch: WorldBatch) -> Dict[str, Tensor]:
        obs = pad_obs(batch.obs, self.cfg.max_obs_dim)
        next_obs = pad_obs(batch.next_obs, self.cfg.max_obs_dim)
        actions = batch.actions.long().view(-1, 1)

        pred = self.core.predict_latent_after_actions(obs, actions, sample_context=True)
        target = self.core.encode_target(next_obs, detach=True)
        z = pred["z_final"]

        return {
            "z_final": z,
            "prior_mu": pred["prior_mu"],
            "prior_logvar": pred["prior_logvar"],
            "target_mu": target["mu"],
            "target_logvar": target["logvar"],
            "context_mu": pred["context_mu"],
            "context_logvar": pred["context_logvar"],
            "pred_next_obs": self.obs_decoder(z),
            "target_next_obs": next_obs,
            "pred_reward": self.reward_head(z).squeeze(-1),
            "target_reward": batch.rewards.float().view(-1),
            "pred_done_logit": self.done_head(z).squeeze(-1),
            "target_done": batch.dones.float().view(-1),
        }

    def init_state(self, obs: Tensor, sample_context: bool = True) -> Dict[str, Tensor]:
        """Initialize a recurrent latent rollout state from an observation.

        Unlike `predict_step`, the returned state carries EFLA memory across future
        calls to `stateful_step`. This is needed for partially observable/memory
        tasks where a fact is visible early and hidden later.
        """

        obs = pad_obs(obs, self.cfg.max_obs_dim)
        enc = self.core.encode_context(obs, sample=sample_context)
        memory = self.core.initial_memory(obs.shape[0], obs.device)
        return {"z": enc["z"], "memory": memory, "context_mu": enc["mu"], "context_logvar": enc["logvar"]}

    def stateful_step(self, state: Dict[str, Tensor], actions: Tensor) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        """Advance a recurrent latent state by one action without re-encoding obs."""

        step = self.core.transition(state["z"], actions.long().view(-1), state["memory"])
        z = step["z_next"]
        next_state = {"z": z, "memory": step["memory_next"], "context_mu": state["context_mu"], "context_logvar": state["context_logvar"]}
        out = {
            "next_obs": self.obs_decoder(z),
            "reward": self.reward_head(z).squeeze(-1),
            "done_logit": self.done_head(z).squeeze(-1),
            "prior_mu": step["prior_mu"],
            "prior_logvar": step["prior_logvar"],
        }
        return next_state, out

    def loss(self, batch: WorldBatch) -> Tuple[Tensor, Dict[str, Tensor]]:
        out = self.forward(batch)

        # Stable latent objective: Gaussian NLL can become strongly negative by
        # collapsing log-variance to its clamp floor. For the Phase 2 MVP we use
        # direct latent mean matching plus a tiny logvar regularizer so the total
        # loss remains interpretable and bounded below by the reconstruction terms.
        latent_mse = F.mse_loss(out["prior_mu"], out["target_mu"])
        latent_logvar_reg = out["prior_logvar"].pow(2).mean()
        kl = diagonal_kl_to_standard_normal(out["context_mu"], out["context_logvar"])
        obs_mse = F.mse_loss(out["pred_next_obs"], out["target_next_obs"])
        reward_mse = F.mse_loss(out["pred_reward"], out["target_reward"])
        done_bce = F.binary_cross_entropy_with_logits(out["pred_done_logit"], out["target_done"])

        total = (
            self.cfg.latent_weight * latent_mse
            + self.cfg.logvar_weight * latent_logvar_reg
            + self.cfg.beta_kl * kl
            + self.cfg.obs_weight * obs_mse
            + self.cfg.reward_weight * reward_mse
            + self.cfg.done_weight * done_bce
        )
        metrics = {
            "total": total.detach(),
            "latent_mse": latent_mse.detach(),
            "latent_logvar_reg": latent_logvar_reg.detach(),
            "kl": kl.detach(),
            "obs_mse": obs_mse.detach(),
            "reward_mse": reward_mse.detach(),
            "done_bce": done_bce.detach(),
        }
        return total, metrics

    @torch.no_grad()
    def predict_step(self, obs: Tensor, actions: Tensor) -> Dict[str, Tensor]:
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
