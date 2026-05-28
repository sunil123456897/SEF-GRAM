from __future__ import annotations

import copy
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
from sef_gram.utils import compute_dirichlet_energy


Tensor = torch.Tensor


@dataclass
class WorldBatch:
    """Common batch format for universal world-model training.
    
    Now expects sequences for block-wise Diffusion-GRAM training.
    Shapes:
    obs: [B, T, D]
    actions: [B, T]
    next_obs: [B, T, D]
    rewards: [B, T]
    dones: [B, T]
    """

    obs: Tensor
    actions: Tensor
    next_obs: Tensor
    rewards: Tensor
    dones: Tensor
    advantages: Optional[Tensor] = None
    env_ids: Optional[Tensor] = None
    task_emb: Optional[Tensor] = None

    def to(self, device: torch.device) -> "WorldBatch":
        return WorldBatch(
            obs=self.obs.to(device),
            actions=self.actions.to(device),
            next_obs=self.next_obs.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            advantages=None if self.advantages is None else self.advantages.to(device),
            env_ids=None if self.env_ids is None else self.env_ids.to(device),
            task_emb=None if self.task_emb is None else self.task_emb.to(device),
        )


@dataclass
class WorldModelConfig:
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    env_vocab_size: int = 100  # Needed for attractors
    beta_kl: float = 1e-3
    latent_weight: float = 1.0
    obs_weight: float = 1.0
    reward_weight: float = 1.0
    done_weight: float = 0.2
    logvar_weight: float = 1e-3
    attractor_weight: float = 1.0  # Weight for Discrete Attractor (Commitment Loss)
    ema_decay: float = 0.99
    block_size_k: int = 4  # Diffusion-GRAM active block size
    use_efla: bool = True


def pad_obs(obs: Tensor, max_obs_dim: int) -> Tensor:
    if obs.shape[-1] == max_obs_dim:
        return obs.float()
    if obs.shape[-1] > max_obs_dim:
        raise ValueError(f"obs dim {obs.shape[-1]} exceeds max_obs_dim={max_obs_dim}")
    pad_width = max_obs_dim - obs.shape[-1]
    return F.pad(obs.float(), (0, pad_width))


class UniversalWorldModel(nn.Module):
    """SEF-GRAM Phase 2 universal latent world-model core.
    
    Upgraded with Diffusion-GRAM (Block-wise EMA training) and EFLA Discrete Attractors.
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
                env_vocab_size=cfg.env_vocab_size,
                recursion_depth=1,
                num_trajectories=1,
                reward_names=("reward",),
                use_efla=cfg.use_efla,
            )
        )
        
        # Diffusion-GRAM: Shadow model for Coordinate Anchoring
        self.ema_core = copy.deepcopy(self.core)
        for param in self.ema_core.parameters():
            param.requires_grad = False
            
        self.obs_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.max_obs_dim),
        )
        self.reward_head = nn.Linear(cfg.latent_dim, 1)
        self.done_head = nn.Linear(cfg.latent_dim, 1)
        self.value_head = nn.Linear(cfg.latent_dim, 1)

    def update_ema(self):
        with torch.no_grad():
            for ema_p, p in zip(self.ema_core.parameters(), self.core.parameters()):
                ema_p.data.mul_(self.cfg.ema_decay).add_(p.data, alpha=1 - self.cfg.ema_decay)

    def forward(self, batch: WorldBatch) -> Dict[str, Tensor]:
        """Block-wise Diffusion-GRAM transition."""
        obs = pad_obs(batch.obs, self.cfg.max_obs_dim)  # [B, T, D]
        next_obs = pad_obs(batch.next_obs, self.cfg.max_obs_dim)
        actions = batch.actions.long() # [B, T]
        
        B, T_seq = actions.shape
        K = min(self.cfg.block_size_k, max(1, T_seq))
        
        # 1. Random start point t
        t_start = torch.randint(0, max(1, T_seq - K + 1), (1,)).item()
        
        # 2. EMA Rollout up to t_start (Coordinate Anchoring)
        with torch.no_grad():
            # Initial encode using obs[:, 0]
            enc_ema = self.ema_core.encode_context(obs[:, 0], sample=False)
            z_ema = enc_ema["z"]
            memory_ema = self.ema_core.initial_memory(B, z_ema.device)
            
            for t in range(t_start):
                step_ema = self.ema_core.transition(z_ema, actions[:, t], memory_ema, task_emb=batch.task_emb)
                z_ema = step_ema["z_next"]
                memory_ema = step_ema["memory_next"]
                
            # EMA Targets for t_start to t_start + K
            z_targets = []
            z_ema_target = z_ema
            memory_ema_target = memory_ema
            for t in range(t_start, t_start + K):
                step_ema = self.ema_core.transition(z_ema_target, actions[:, t], memory_ema_target, task_emb=batch.task_emb)
                z_ema_target = step_ema["z_next"]
                memory_ema_target = step_ema["memory_next"]
                z_targets.append(z_ema_target)
            z_targets = torch.stack(z_targets, dim=1) # [B, K, latent_dim]
            
        # 3. Active Rollout for K steps
        z_active = z_ema.detach() # Anchor
        memory_active = memory_ema.detach()
        
        z_preds = []
        prior_mus = []
        prior_logvars = []
        
        for t in range(t_start, t_start + K):
            step_active = self.core.transition(z_active, actions[:, t], memory_active, task_emb=batch.task_emb)
            z_active = step_active["z_next"]
            memory_active = step_active["memory_next"]
            z_preds.append(z_active)
            prior_mus.append(step_active["prior_mu"])
            prior_logvars.append(step_active["prior_logvar"])
            
        z_preds = torch.stack(z_preds, dim=1)
        prior_mus = torch.stack(prior_mus, dim=1)
        prior_logvars = torch.stack(prior_logvars, dim=1)
        
        # VJEPA Target Encoding
        target_enc = self.core.encode_target(next_obs[:, t_start:t_start+K], detach=True)
        
        # Catastrophic Interference Protection: Policy reads detached latents!
        policy_logits = self.core.policy_head(z_preds.detach())
        
        return {
            "z_preds": z_preds,                 # [B, K, D]
            "z_targets": z_targets,             # [B, K, D]
            "prior_mus": prior_mus,
            "prior_logvars": prior_logvars,
            "target_mus": target_enc["mu"],
            "target_logvars": target_enc["logvar"],
            "context_mu": enc_ema["mu"].unsqueeze(1).expand(-1, K, -1),
            "context_logvar": enc_ema["logvar"].unsqueeze(1).expand(-1, K, -1),
            "pred_next_obs": self.obs_decoder(z_preds),
            "target_next_obs": next_obs[:, t_start:t_start+K],
            "pred_reward": self.reward_head(z_preds).squeeze(-1),
            "target_reward": batch.rewards[:, t_start:t_start+K].float(),
            "pred_done_logit": self.done_head(z_preds).squeeze(-1),
            "target_done": batch.dones[:, t_start:t_start+K].float(),
            "pred_value": self.value_head(z_preds).squeeze(-1),
            "policy_logits": policy_logits,
            "t_start": t_start,
            "K": K,
        }

    def loss(self, batch: WorldBatch, lambda_gdpo: float = 0.0) -> Tuple[Tensor, Dict[str, Tensor]]:
        out = self.forward(batch)
        
        # VJEPA loss
        latent_mse = F.mse_loss(out["prior_mus"], out["target_mus"])
        latent_logvar_reg = out["prior_logvars"].pow(2).mean()
        kl = diagonal_kl_to_standard_normal(out["context_mu"], out["context_logvar"])
        
        # Output decoders
        obs_mse = F.mse_loss(out["pred_next_obs"], out["target_next_obs"])
        reward_mse = F.mse_loss(out["pred_reward"], out["target_reward"])
        done_bce = F.binary_cross_entropy_with_logits(out["pred_done_logit"], out["target_done"])
        
        with torch.no_grad():
            value_target = out["target_reward"] + 0.95 * out["pred_value"].detach() * (1.0 - out["target_done"])
        value_mse = F.mse_loss(out["pred_value"], value_target)
        
        # EFLA Discrete Attractor Loss (VQ-VAE Commitment Loss style)
        attractor_loss = torch.tensor(0.0, device=latent_mse.device)
        active_attractors_count = 0.0
        
        if self.core.env_head is not None:
            # env_head.weight: [vocab_size, latent_dim]
            vocab_emb = self.core.env_head.weight.detach()
            z_flat = out["z_preds"].reshape(-1, self.cfg.latent_dim)
            
            dists = torch.cdist(z_flat, vocab_emb) # [B*K, vocab_size]
            nearest_idx = torch.argmin(dists, dim=-1)
            nearest_emb = vocab_emb[nearest_idx]
            
            # Loss pulls state toward nearest embedding (stop_grad on embedding)
            attractor_loss = F.mse_loss(z_flat, nearest_emb.detach())
            
            unique_attractors = torch.unique(nearest_idx).numel()
            active_attractors_count = unique_attractors / vocab_emb.shape[0]

        # Latent Continuity Check
        latent_continuity_cosine = F.cosine_similarity(
            out["z_preds"].reshape(-1, self.cfg.latent_dim), 
            out["z_targets"].reshape(-1, self.cfg.latent_dim)
        ).mean()

        total_echo = (
            self.cfg.latent_weight * latent_mse
            + self.cfg.logvar_weight * latent_logvar_reg
            + self.cfg.beta_kl * kl
            + self.cfg.obs_weight * obs_mse
            + self.cfg.reward_weight * reward_mse
            + self.cfg.done_weight * done_bce
            + 0.1 * value_mse
            + self.cfg.attractor_weight * attractor_loss
        )
        
        # GDPO Loss (RL)
        gdpo_loss = torch.tensor(0.0, device=total_echo.device)
        if batch.advantages is not None:
            acts = batch.actions[:, out["t_start"]:out["t_start"]+out["K"]].long()
            log_probs = F.log_softmax(out["policy_logits"], dim=-1).gather(-1, acts.unsqueeze(-1)).squeeze(-1)
            advs = batch.advantages[:, out["t_start"]:out["t_start"]+out["K"]].to(total_echo.device)
            gdpo_loss = -(log_probs * advs).mean()

        # Dynamic Schedule
        total = (1.0 - lambda_gdpo) * total_echo + lambda_gdpo * gdpo_loss
        
        dirichlet_energy = compute_dirichlet_energy(out["z_preds"][:, -1])
        
        metrics = {
            "total": total.detach(),
            "echo_loss": total_echo.detach(),
            "gdpo_loss": gdpo_loss.detach(),
            "latent_mse": latent_mse.detach(),
            "obs_mse": obs_mse.detach(),
            "attractor_loss": attractor_loss.detach(),
            "active_attractors_count": torch.tensor(active_attractors_count, device=total.device),
            "latent_continuity_cosine": latent_continuity_cosine.detach(),
            "dirichlet_energy": dirichlet_energy.detach(),
        }
        
        # Update EMA
        self.update_ema()
        
        return total, metrics

    # Predict methods adjusted for 1-step logic remaining compatible
    @torch.no_grad()
    def predict_step(self, obs: Tensor, actions: Tensor) -> Dict[str, Tensor]:
        # For legacy 1-step usage in inference
        dummy = WorldBatch(
            obs=obs.unsqueeze(1),
            actions=actions.unsqueeze(1),
            next_obs=torch.zeros(obs.shape[0], 1, self.cfg.max_obs_dim, device=obs.device),
            rewards=torch.zeros(obs.shape[0], 1, device=obs.device),
            dones=torch.zeros(obs.shape[0], 1, device=obs.device),
        )
        out = self.forward(dummy)
        return {
            "next_obs": out["pred_next_obs"][:, 0],
            "reward": out["pred_reward"][:, 0],
            "done_prob": torch.sigmoid(out["pred_done_logit"][:, 0]),
        }
