from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


@dataclass
class SEFGRAMConfig:
    """Configuration for the integrated SEF-GRAM research system.

    The defaults are deliberately small so the system can be smoke-tested on CPU.
    Increase latent_dim, hidden_dim, recursion_depth, and num_trajectories for real experiments.
    """

    input_dim: int = 32
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    env_vocab_size: int = 0
    recursion_depth: int = 4
    num_trajectories: int = 4
    reward_names: Tuple[str, ...] = ("correctness", "format", "brevity")
    min_logvar: float = -8.0
    max_logvar: float = 4.0


def reparameterize(mu: Tensor, logvar: Tensor, sample: bool = True) -> Tensor:
    if not sample:
        return mu
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std


def diagonal_gaussian_nll(target: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
    """Mean negative log likelihood of target under diagonal Gaussian N(mu, exp(logvar))."""

    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar)).sum(dim=-1).mean()


def diagonal_kl_to_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL[N(mu, var) || N(0, I)] averaged over batch."""

    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()


def gaussian_product_of_experts(
    mu_a: Tensor,
    logvar_a: Tensor,
    mu_b: Tensor,
    logvar_b: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Fuse two diagonal Gaussian experts by precision addition."""

    precision_a = torch.exp(-logvar_a)
    precision_b = torch.exp(-logvar_b)
    precision = precision_a + precision_b
    mu = (mu_a * precision_a + mu_b * precision_b) / (precision + eps)
    logvar = -torch.log(precision + eps)
    return mu, logvar


class VariationalLatentEncoder(nn.Module):
    """VJEPA-style latent encoder returning a diagonal Gaussian belief state."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, min_logvar: float, max_logvar: float):
        super().__init__()
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
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
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(self.min_logvar, self.max_logvar)
        return mu, logvar


class ExactEFLACell(nn.Module):
    """EFLA-inspired exact continuous-time rank-1 fast-weight update.

    This implements the analytically integrated rank-1 update used by the rest of the
    system. It is intentionally written as a transparent PyTorch module instead of a
    fused kernel: correctness and inspectability are more important for the first MVP.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.key = nn.Linear(latent_dim, latent_dim, bias=False)
        self.value = nn.Linear(latent_dim, latent_dim, bias=False)
        self.in_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.state_norm = nn.LayerNorm(latent_dim)
        self.log_beta = nn.Parameter(torch.tensor(-2.0))

    def _alpha(self, k: Tensor, eps: float = 1e-8) -> Tensor:
        k2 = torch.sum(k * k, dim=-1, keepdim=True)
        beta = F.softplus(self.log_beta) + eps
        x = beta * k2
        taylor = beta * (1.0 - 0.5 * x + (x * x) / 6.0)
        exact = -torch.expm1(-x) / (k2 + eps)
        return torch.where(x.abs() < 1e-4, taylor, exact)

    def forward(self, z_prev: Tensor, memory_prev: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.in_proj(z_prev)
        k = self.key(x)
        v = self.value(x)

        retrieved = torch.bmm(memory_prev, k.unsqueeze(-1)).squeeze(-1)
        residual = v - retrieved
        alpha = self._alpha(k)

        update = alpha.unsqueeze(-1) * torch.bmm(residual.unsqueeze(-1), k.unsqueeze(1))
        memory_next = memory_prev + update
        z_memory = torch.bmm(memory_next, k.unsqueeze(-1)).squeeze(-1)
        z_next = self.state_norm(z_prev + self.out_proj(z_memory))
        return z_next, memory_next


class ActionConditionedTransition(nn.Module):
    """Action-conditioned latent transition with EFLA memory dynamics."""

    def __init__(self, cfg: SEFGRAMConfig):
        super().__init__()
        self.cfg = cfg
        self.action_embedding = nn.Embedding(cfg.num_actions, cfg.latent_dim)
        self.pre = nn.Sequential(
            nn.Linear(cfg.latent_dim * 2, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )
        self.cell = ExactEFLACell(cfg.latent_dim)
        self.prior_mu = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        self.prior_logvar = nn.Linear(cfg.latent_dim, cfg.latent_dim)

    def forward(self, z: Tensor, action_ids: Tensor, memory: Tensor) -> Dict[str, Tensor]:
        action_vec = self.action_embedding(action_ids.long())
        z_action = self.pre(torch.cat([z, action_vec], dim=-1))
        z_next, memory_next = self.cell(z_action, memory)
        return {
            "z_next": z_next,
            "memory_next": memory_next,
            "prior_mu": self.prior_mu(z_next),
            "prior_logvar": self.prior_logvar(z_next).clamp(self.cfg.min_logvar, self.cfg.max_logvar),
        }


class StochasticRecursiveWorldModel(nn.Module):
    """Integrated SEF-GRAM core.

    Components:
    - VJEPA-like variational context and target encoders.
    - EFLA latent memory transition.
    - GRAM-like stochastic recursive multi-trajectory rollout.
    - Policy/reward heads for RL and optional ECHO environment-token prediction.
    """

    def __init__(self, cfg: SEFGRAMConfig):
        super().__init__()
        self.cfg = cfg
        self.context_encoder = VariationalLatentEncoder(
            cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.min_logvar, cfg.max_logvar
        )
        self.target_encoder = VariationalLatentEncoder(
            cfg.input_dim, cfg.latent_dim, cfg.hidden_dim, cfg.min_logvar, cfg.max_logvar
        )
        self.transition = ActionConditionedTransition(cfg)
        self.policy_head = nn.Linear(cfg.latent_dim, cfg.num_actions)
        self.reward_heads = nn.ModuleDict({name: nn.Linear(cfg.latent_dim, 1) for name in cfg.reward_names})
        self.env_head = nn.Linear(cfg.latent_dim, cfg.env_vocab_size) if cfg.env_vocab_size > 0 else None

    def initial_memory(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.cfg.latent_dim, self.cfg.latent_dim, device=device)

    def encode_context(self, obs: Tensor, sample: bool = True) -> Dict[str, Tensor]:
        mu, logvar = self.context_encoder(obs)
        z = reparameterize(mu, logvar, sample=sample)
        return {"z": z, "mu": mu, "logvar": logvar}

    def encode_target(self, obs: Tensor, detach: bool = True) -> Dict[str, Tensor]:
        mu, logvar = self.target_encoder(obs)
        if detach:
            mu = mu.detach()
            logvar = logvar.detach()
        return {"mu": mu, "logvar": logvar}

    def policy(self, z: Tensor) -> Tensor:
        return self.policy_head(z)

    def reward_predictions(self, z: Tensor) -> Dict[str, Tensor]:
        return {name: head(z).squeeze(-1) for name, head in self.reward_heads.items()}

    def predict_latent_after_actions(
        self,
        context_obs: Tensor,
        action_ids: Tensor,
        sample_context: bool = True,
    ) -> Dict[str, Tensor]:
        """Predict final latent Gaussian after a provided action sequence.

        action_ids shape: (B, T).
        """

        enc = self.encode_context(context_obs, sample=sample_context)
        z = enc["z"]
        memory = self.initial_memory(z.shape[0], z.device)
        prior_mu, prior_logvar = enc["mu"], enc["logvar"]
        for t in range(action_ids.shape[1]):
            step = self.transition(z, action_ids[:, t], memory)
            z = step["z_next"]
            memory = step["memory_next"]
            prior_mu = step["prior_mu"]
            prior_logvar = step["prior_logvar"]
        return {
            "context_mu": enc["mu"],
            "context_logvar": enc["logvar"],
            "z_final": z,
            "memory_final": memory,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "policy_logits": self.policy(z),
            "reward_preds": self.reward_predictions(z),
        }

    def recursive_rollout(
        self,
        context_obs: Tensor,
        depth: Optional[int] = None,
        num_trajectories: Optional[int] = None,
        action_ids: Optional[Tensor] = None,
        sample_actions: bool = True,
    ) -> Dict[str, Tensor]:
        """Run stochastic multi-trajectory latent recursion.

        Returns tensors with shape (B, K, T, ...). If action_ids is provided it may be
        (B, T) or (B, K, T); otherwise actions are sampled or greedily chosen from the policy.
        """

        depth = depth or self.cfg.recursion_depth
        num_trajectories = num_trajectories or self.cfg.num_trajectories
        B = context_obs.shape[0]
        device = context_obs.device

        enc = self.encode_context(context_obs, sample=True)
        z = enc["z"].unsqueeze(1).expand(B, num_trajectories, self.cfg.latent_dim).reshape(B * num_trajectories, -1)
        memory = self.initial_memory(B * num_trajectories, device)

        all_z = []
        all_actions = []
        all_logits = []
        all_reward_preds = {name: [] for name in self.cfg.reward_names}

        for t in range(depth):
            logits = self.policy(z)
            if action_ids is None:
                if sample_actions:
                    action = torch.distributions.Categorical(logits=logits).sample()
                else:
                    action = logits.argmax(dim=-1)
            else:
                if action_ids.dim() == 2:
                    action = action_ids[:, t].unsqueeze(1).expand(B, num_trajectories).reshape(-1)
                elif action_ids.dim() == 3:
                    action = action_ids[:, :, t].reshape(-1)
                else:
                    raise ValueError("action_ids must have shape (B, T) or (B, K, T)")

            step = self.transition(z, action, memory)
            z = step["z_next"]
            memory = step["memory_next"]
            rewards = self.reward_predictions(z)

            all_z.append(z.view(B, num_trajectories, self.cfg.latent_dim))
            all_actions.append(action.view(B, num_trajectories))
            all_logits.append(logits.view(B, num_trajectories, self.cfg.num_actions))
            for name, pred in rewards.items():
                all_reward_preds[name].append(pred.view(B, num_trajectories))

        return {
            "latents": torch.stack(all_z, dim=2),
            "actions": torch.stack(all_actions, dim=2),
            "policy_logits": torch.stack(all_logits, dim=2),
            "reward_preds": {name: torch.stack(vals, dim=2) for name, vals in all_reward_preds.items()},
            "context_mu": enc["mu"],
            "context_logvar": enc["logvar"],
        }

    def echo_logits_from_latents(self, latents: Tensor) -> Tensor:
        if self.env_head is None:
            raise RuntimeError("env_vocab_size must be > 0 to use ECHO environment-token prediction")
        return self.env_head(latents)


class SEFGRAMObjective:
    """Loss functions for VJEPA, GRAM rollouts, ECHO, and GDPO."""

    @staticmethod
    def vjepa_latent_loss(
        model: StochasticRecursiveWorldModel,
        context_obs: Tensor,
        target_obs: Tensor,
        action_ids: Tensor,
        beta_kl: float = 1e-3,
        collapse_margin: float = 1e-2,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        pred = model.predict_latent_after_actions(context_obs, action_ids, sample_context=True)
        target = model.encode_target(target_obs, detach=True)

        pred_nll = diagonal_gaussian_nll(target["mu"], pred["prior_mu"], pred["prior_logvar"])
        kl = diagonal_kl_to_standard_normal(pred["context_mu"], pred["context_logvar"])
        latent_variance = pred["z_final"].var(dim=0, unbiased=False).mean()
        collapse_penalty = F.relu(collapse_margin - latent_variance)
        total = pred_nll + beta_kl * kl + collapse_penalty
        return total, {
            "total": total.detach(),
            "pred_nll": pred_nll.detach(),
            "kl": kl.detach(),
            "collapse_penalty": collapse_penalty.detach(),
            "latent_variance": latent_variance.detach(),
        }

    @staticmethod
    def gdpo_advantages(
        rewards: Dict[str, Iterable[float] | Tensor],
        weights: Optional[Dict[str, float]] = None,
        eps: float = 1e-8,
    ) -> Tensor:
        advantages = None
        weights = weights or {}
        for name, values in rewards.items():
            tensor = values if torch.is_tensor(values) else torch.tensor(list(values), dtype=torch.float32)
            tensor = tensor.float()
            std = tensor.std(unbiased=False)
            if tensor.numel() <= 1 or std < eps:
                normed = torch.zeros_like(tensor)
            else:
                normed = (tensor - tensor.mean()) / (std + eps)
            normed = normed * float(weights.get(name, 1.0))
            advantages = normed if advantages is None else advantages + normed

        if advantages is None:
            raise ValueError("rewards must contain at least one objective")
        final_std = advantages.std(unbiased=False)
        if advantages.numel() > 1 and final_std >= eps:
            advantages = (advantages - advantages.mean()) / (final_std + eps)
        return advantages

    @staticmethod
    def echo_gdpo_loss(
        policy_logits: Tensor,
        actions: Tensor,
        advantages: Tensor,
        env_logits: Optional[Tensor] = None,
        env_targets: Optional[Tensor] = None,
        echo_lambda: float = 0.05,
        entropy_coef: float = 0.0,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        B, T, A = policy_logits.shape
        log_probs = F.log_softmax(policy_logits, dim=-1).gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        policy_loss = -(log_probs.sum(dim=-1) * advantages.to(policy_logits.device)).mean()

        probs = F.softmax(policy_logits, dim=-1)
        entropy = -(probs * F.log_softmax(policy_logits, dim=-1)).sum(dim=-1).mean()
        total = policy_loss - entropy_coef * entropy

        env_loss = torch.tensor(0.0, device=policy_logits.device)
        if env_logits is not None or env_targets is not None:
            if env_logits is None or env_targets is None:
                raise ValueError("env_logits and env_targets must be provided together")
            env_loss = F.cross_entropy(env_logits.reshape(-1, env_logits.shape[-1]), env_targets.reshape(-1).long())
            total = total + echo_lambda * env_loss

        return total, {
            "total": total.detach(),
            "policy_loss": policy_loss.detach(),
            "env_loss": env_loss.detach(),
            "entropy": entropy.detach(),
        }


class LatentPoEPlanner(nn.Module):
    """BJEPA-style latent planner using Product-of-Experts scoring over sampled rollouts."""

    def __init__(self, model: StochasticRecursiveWorldModel):
        super().__init__()
        self.model = model

    @staticmethod
    def fuse(mu_a: Tensor, logvar_a: Tensor, mu_b: Tensor, logvar_b: Tensor) -> Tuple[Tensor, Tensor]:
        return gaussian_product_of_experts(mu_a, logvar_a, mu_b, logvar_b)

    def plan(
        self,
        context_obs: Tensor,
        goal_mu: Optional[Tensor] = None,
        goal_logvar: Optional[Tensor] = None,
        depth: Optional[int] = None,
        num_trajectories: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        rollout = self.model.recursive_rollout(
            context_obs,
            depth=depth,
            num_trajectories=num_trajectories,
            sample_actions=True,
        )
        final_z = rollout["latents"][:, :, -1, :]

        if goal_mu is None:
            reward_score = torch.zeros(final_z.shape[:2], device=final_z.device)
            for preds in rollout["reward_preds"].values():
                reward_score = reward_score + preds[:, :, -1]
            score = reward_score
        else:
            if goal_logvar is None:
                goal_logvar = torch.zeros_like(goal_mu)
            while goal_mu.dim() < final_z.dim():
                goal_mu = goal_mu.unsqueeze(1)
                goal_logvar = goal_logvar.unsqueeze(1)
            score = -0.5 * ((final_z - goal_mu).pow(2) * torch.exp(-goal_logvar)).sum(dim=-1)

        best_idx = score.argmax(dim=1)
        batch_idx = torch.arange(context_obs.shape[0], device=context_obs.device)
        best_actions = rollout["actions"][batch_idx, best_idx]
        return {"best_actions": best_actions, "scores": score, "rollout": rollout}


def build_tiny_system(**overrides) -> StochasticRecursiveWorldModel:
    cfg = SEFGRAMConfig(**overrides)
    return StochasticRecursiveWorldModel(cfg)
