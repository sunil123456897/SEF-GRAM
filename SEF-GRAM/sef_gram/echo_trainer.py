from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sef_gram.full_system import (
    SEFGRAMConfig,
    SEFGRAMObjective,
    StochasticRecursiveWorldModel,
    diagonal_gaussian_nll,
    reparameterize,
)
from sef_gram.turn_based_env import MiniTerminalEnv, TurnBatch
from sef_gram.utils import compute_dirichlet_energy

Tensor = torch.Tensor


@dataclass
class ECHOGDPOConfig:
    steps: int = 500
    batch_size: int = 32
    num_candidates: int = 4
    latent_dim: int = 64
    hidden_dim: int = 128
    max_turns: int = 6
    echo_lambda: float = 0.05
    entropy_coef: float = 0.01
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adamw_lr: float = 3e-4
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class EMALatentEncoder(nn.Module):
    """VJEPA-style EMA-updated target encoder for environment observations.

    Encodes environment responses into a latent Gaussian distribution.
    Updated via exponential moving average of online encoder weights.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, ema_decay: float = 0.999):
        super().__init__()
        self.ema_decay = ema_decay
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.trunk(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 4.0)

    @torch.no_grad()
    def update_ema(self, online: nn.Module) -> None:
        for target_p, online_p in zip(self.parameters(), online.parameters()):
            target_p.data.mul_(self.ema_decay).add_(online_p.data, alpha=1.0 - self.ema_decay)


class ECHOGRAMWrapper(nn.Module):
    """Wrapper with Latent ECHO: predicts latent env states via EMA target encoder.

    Instead of classifying discrete env tokens, this predicts the latent
    representation of environment responses using a VJEPA-style EMA target encoder.
    The env_loss is Gaussian NLL in latent space — noise-invariant world modeling.
    """

    def __init__(self, obs_dim: int, num_actions: int, env_vocab_size: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.num_actions = num_actions
        self.latent_dim = latent_dim

        cfg = SEFGRAMConfig(
            input_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            env_vocab_size=0,
            recursion_depth=1,
            num_trajectories=1,
            reward_names=("task",),
        )
        self.core = StochasticRecursiveWorldModel(cfg)
        self.obj = SEFGRAMObjective()

        self.env_latent_dim = 32
        self.env_target_encoder = EMALatentEncoder(obs_dim, self.env_latent_dim, hidden_dim)
        self.env_online_encoder = EMALatentEncoder(obs_dim, self.env_latent_dim, hidden_dim)

        self.env_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.env_pred_mu = nn.Linear(hidden_dim, self.env_latent_dim)
        self.env_pred_logvar = nn.Linear(hidden_dim, self.env_latent_dim)

    def initial_state(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        flat_obs = obs.reshape(-1, obs.shape[-1])
        enc = self.core.encode_context(flat_obs, sample=True)
        memory = self.core.initial_memory(flat_obs.shape[0], flat_obs.device)
        return enc["z"], memory, enc["mu"]

    def forward_turn(
        self, z: Tensor, memory: Tensor, actions: Tensor, obs: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        step = self.core.transition(z, actions, memory)
        return step["z_next"], step["memory_next"]

    def policy_logits(self, z: Tensor) -> Tensor:
        return self.core.policy_head(z)

    def encode_env_target(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            return self.env_target_encoder(obs)

    def predict_env_latent(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.env_predictor(z)
        mu = self.env_pred_mu(h)
        logvar = self.env_pred_logvar(h).clamp(-8.0, 4.0)
        return mu, logvar

    def env_logits_from_z(self, z: Tensor) -> Optional[Tensor]:
        return None

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        self.env_target_encoder.update_ema(self.env_online_encoder)


def _gdpo_advantages_from_rewards(rewards: Tensor, eps: float = 1e-8) -> Tensor:
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True, unbiased=False)
    normed = torch.zeros_like(rewards)
    valid = std.squeeze(-1) >= eps
    if valid.ndim == 0:
        valid = valid.unsqueeze(0)
    if valid.any():
        normed[valid] = (rewards[valid] - mean[valid]) / (std[valid] + eps)
    return normed


def run_echo_gdpo_step(
    model: ECHOGRAMWrapper,
    env: MiniTerminalEnv,
    cfg: ECHOGDPOConfig,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    B = cfg.batch_size
    K = cfg.num_candidates

    batch = env.reset(B, device, num_candidates=K)
    obs = batch.obs
    z, memory, _ = model.initial_state(obs)

    all_policy_logits: list[Tensor] = []
    all_actions: list[Tensor] = []
    all_env_mu: list[Tensor] = []
    all_env_logvar: list[Tensor] = []
    all_target_mu: list[Tensor] = []
    all_target_logvar: list[Tensor] = []
    all_rewards: list[Tensor] = []

    for _ in range(cfg.max_turns):
        logits = model.policy_logits(z)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = dist.sample()

        actions_bk = actions_flat.view(B, K)
        env_batch = env.step(actions_bk)

        z_r = z.view(B, K, -1)
        z_avg = z_r.mean(dim=1)

        pred_mu, pred_logvar = model.predict_env_latent(z_avg)
        next_obs_flat = env_batch.obs.reshape(-1, env_batch.obs.shape[-1])
        target_mu, target_logvar = model.encode_env_target(next_obs_flat.view(B, K, -1).mean(dim=1))

        z_next, memory_next = model.forward_turn(z, memory, actions_flat)

        all_policy_logits.append(logits)
        all_actions.append(actions_flat)
        all_env_mu.append(pred_mu)
        all_env_logvar.append(pred_logvar)
        all_target_mu.append(target_mu)
        all_target_logvar.append(target_logvar)
        all_rewards.append(env_batch.reward)

        z = z_next
        memory = memory_next

        if env_batch.done.any():
            next_obs = env_batch.obs.reshape(-1, env_batch.obs.shape[-1])
            enc = model.core.encode_context(next_obs, sample=True)
            z_done = enc["z"]
            memory_done = model.core.initial_memory(z_done.shape[0], device)
            curr_done = env_batch.done.reshape(-1)
            z = torch.where(curr_done.unsqueeze(-1), z_done, z)
            memory = torch.where(curr_done.view(-1, 1, 1), memory_done, memory)

    actions_t = torch.stack(all_actions, dim=1)
    logits_t = torch.stack(all_policy_logits, dim=1)
    rewards_t = torch.stack(all_rewards, dim=-1)

    traj_rewards = rewards_t.sum(dim=-1)
    advantages_expanded = _gdpo_advantages_from_rewards(traj_rewards)
    advantages = advantages_expanded.reshape(-1)

    log_probs = F.log_softmax(logits_t, dim=-1).gather(-1, actions_t.long().unsqueeze(-1)).squeeze(-1)
    policy_loss = -(log_probs.sum(dim=-1) * advantages.to(logits_t.device)).mean()

    probs = F.softmax(logits_t, dim=-1)
    entropy = -(probs * F.log_softmax(logits_t, dim=-1)).sum(dim=-1).mean()

    env_mu_t = torch.stack(all_env_mu, dim=1)
    env_logvar_t = torch.stack(all_env_logvar, dim=1)
    target_mu_t = torch.stack(all_target_mu, dim=1).detach()
    target_logvar_t = torch.stack(all_target_logvar, dim=1).detach()

    env_loss = diagonal_gaussian_nll(
        target_mu_t.reshape(-1, model.env_latent_dim),
        env_mu_t.reshape(-1, model.env_latent_dim),
        env_logvar_t.reshape(-1, model.env_latent_dim),
    )

    total = policy_loss + cfg.echo_lambda * env_loss - cfg.entropy_coef * entropy

    de = compute_dirichlet_energy(z.view(-1, z.shape[-1]))
    metrics = {
        "total": total.detach(),
        "policy_loss": policy_loss.detach(),
        "env_loss": env_loss.detach(),
        "entropy": entropy.detach(),
        "dirichlet_energy": de.detach(),
        "mean_reward": traj_rewards.mean().detach(),
        "success_rate": (traj_rewards.max(dim=-1).values > 0.5).float().mean().detach(),
    }
    return total, metrics


def train_echo_gdpo(cfg: ECHOGDPOConfig) -> ECHOGRAMWrapper:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    env = MiniTerminalEnv(seed=cfg.seed)
    model = ECHOGRAMWrapper(
        obs_dim=env.OBS_DIM,
        num_actions=env.NUM_ACTIONS,
        env_vocab_size=env.NUM_ENV_CLASSES,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    ).to(device)

    from sef_gram.optimization import MuonWithAuxAdam

    optimizer = MuonWithAuxAdam(
        model.parameters(),
        lr=cfg.muon_lr,
        momentum=cfg.muon_momentum,
        ns_steps=cfg.muon_ns_steps,
        adamw_lr=cfg.adamw_lr,
        adamw_betas=cfg.adamw_betas,
        adamw_wd=cfg.adamw_wd,
    )

    print(
        f"[echo_gdpo] train steps={cfg.steps} batch={cfg.batch_size} "
        f"candidates={cfg.num_candidates} device={device}"
    )

    for step in range(1, cfg.steps + 1):
        model.train()
        optimizer.zero_grad()
        loss, metrics = run_echo_gdpo_step(model, env, cfg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_target_encoder()

        if step == 1 or step % max(1, cfg.steps // 10) == 0:
            print(
                f"[echo_gdpo] step={step:04d} "
                f"total={float(loss.item()):.4f} "
                f"policy={float(metrics['policy_loss'].item()):.4f} "
                f"env={float(metrics['env_loss'].item()):.4f} "
                f"entropy={float(metrics['entropy'].item()):.4f} "
                f"success={float(metrics['success_rate'].item()):.3f} "
                f"avg_r={float(metrics['mean_reward'].item()):.3f} "
                f"E_Dirichlet={float(metrics['dirichlet_energy'].item()):.4f}"
            )

    return model
