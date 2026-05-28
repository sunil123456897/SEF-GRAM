from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Protocol, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.world_model_rollout_eval import evaluate_true_rollout
from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.world_envs import build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldBatch, WorldModelConfig
from sef_gram.world_rollout_envs import rollout_envs


class TrainableWorldModel(Protocol):
    def loss(self, batch: WorldBatch): ...
    def forward(self, batch: WorldBatch) -> Dict[str, torch.Tensor]: ...
    def predict_step(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]: ...


@dataclass
class MultiStepTrainConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    rollout_horizon: int = 5
    rollout_loss_weight: float = 1.0
    one_step_loss_weight: float = 1.0
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 29
    export_csv: str = ""


def differentiable_predict_step(
    model: TrainableWorldModel,
    obs: torch.Tensor,
    actions: torch.Tensor,
    max_obs_dim: int,
) -> Dict[str, torch.Tensor]:
    dummy = WorldBatch(
        obs=obs,
        actions=actions,
        next_obs=torch.zeros(obs.shape[0], max_obs_dim, device=obs.device),
        rewards=torch.zeros(obs.shape[0], device=obs.device),
        dones=torch.zeros(obs.shape[0], device=obs.device),
    )
    out = model.forward(dummy)
    if "pred_next_obs" not in out:
        raise RuntimeError("model.forward must return pred_next_obs")
    return {
        "next_obs": out["pred_next_obs"],
        "reward": out["pred_reward"],
        "done_logit": out["pred_done_logit"],
    }


def rollout_loss_for_env(
    model: TrainableWorldModel,
    env,
    cfg: MultiStepTrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    true_obs = env.reset(cfg.batch_size, device)
    pred_obs = true_obs
    obs_losses = []
    reward_losses = []
    done_losses = []

    for _ in range(cfg.rollout_horizon):
        actions = env.sample_actions(cfg.batch_size, device)
        true_step = env.step_from_obs(true_obs, actions)
        pred = differentiable_predict_step(model, pred_obs, actions, cfg.max_obs_dim)

        pred_next_obs = pred["next_obs"]
        obs_losses.append(F.mse_loss(pred_next_obs[:, : env.obs_dim], true_step.next_obs[:, : env.obs_dim]))
        reward_losses.append(F.mse_loss(pred["reward"], true_step.rewards))
        done_losses.append(F.binary_cross_entropy_with_logits(pred["done_logit"], true_step.dones))

        true_obs = true_step.next_obs.detach()
        pred_obs = pred_next_obs

    obs_loss = torch.stack(obs_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()
    done_loss = torch.stack(done_losses).mean()
    total = obs_loss + reward_loss + 0.2 * done_loss
    return total, {"rollout_obs_mse": obs_loss.detach(), "rollout_reward_mse": reward_loss.detach(), "rollout_done_bce": done_loss.detach()}


def multi_env_rollout_loss(
    model: TrainableWorldModel,
    cfg: MultiStepTrainConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    losses = []
    obs_losses = []
    reward_losses = []
    done_losses = []
    for env in rollout_envs(max_obs_dim=cfg.max_obs_dim).values():
        loss, metrics = rollout_loss_for_env(model, env, cfg, device)
        losses.append(loss)
        obs_losses.append(metrics["rollout_obs_mse"])
        reward_losses.append(metrics["rollout_reward_mse"])
        done_losses.append(metrics["rollout_done_bce"])
    total = torch.stack(losses).mean()
    return total, {
        "rollout_total": total.detach(),
        "rollout_obs_mse": torch.stack(obs_losses).mean().detach(),
        "rollout_reward_mse": torch.stack(reward_losses).mean().detach(),
        "rollout_done_bce": torch.stack(done_losses).mean().detach(),
    }


def train_multistep(cfg: MultiStepTrainConfig) -> UniversalWorldModel:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    model = UniversalWorldModel(
        WorldModelConfig(
            max_obs_dim=cfg.max_obs_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.num_actions,
        )
    ).to(device)
    optimizer = MuonWithAuxAdam(
        model.parameters(),
        lr=0.02,
        momentum=0.95,
        ns_steps=5,
        adamw_lr=cfg.lr,
        adamw_betas=(0.9, 0.95),
        adamw_wd=1e-4,
    )

    print(
        f"[sef_gram_multistep] train steps={cfg.steps} batch={cfg.batch_size} "
        f"horizon={cfg.rollout_horizon} device={device}"
    )
    for step in range(1, cfg.steps + 1):
        one_step_batch = batcher.sample(cfg.batch_size, device)
        model.train()
        optimizer.zero_grad()
        one_step_loss, one_step_metrics = model.loss(one_step_batch)
        rollout_loss, rollout_metrics = multi_env_rollout_loss(model, cfg, device)
        total = cfg.one_step_loss_weight * one_step_loss + cfg.rollout_loss_weight * rollout_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.steps // 5) == 0:
            de = one_step_metrics.get("dirichlet_energy", torch.tensor(-1.0))
            print(
                f"[sef_gram_multistep] step={step:04d} total={float(total.item()):.4f} "
                f"one={float(one_step_metrics['total'].item()):.4f} "
                f"rollout={float(rollout_metrics['rollout_total'].item()):.4f} "
                f"roll_obs={float(rollout_metrics['rollout_obs_mse'].item()):.4f} "
                f"roll_reward={float(rollout_metrics['rollout_reward_mse'].item()):.4f} "
                f"E_Dirichlet={float(de.item()):.4f}"
            )
    return model


def run(cfg: MultiStepTrainConfig) -> List[Dict[str, float]]:
    model = train_multistep(cfg)
    rows = evaluate_true_rollout(model, cfg, "sef_gram_multistep")
    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> MultiStepTrainConfig:
    parser = argparse.ArgumentParser(description="Train SEF-GRAM world model with differentiable multi-step rollout loss")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--one-step-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--export-csv", type=str, default="")
    return MultiStepTrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
