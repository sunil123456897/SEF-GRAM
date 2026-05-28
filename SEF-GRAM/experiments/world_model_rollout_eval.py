from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_envs import build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldBatch, WorldModelConfig
from sef_gram.world_rollout_envs import rollout_envs


class WorldModelLike(Protocol):
    def loss(self, batch: WorldBatch): ...
    def predict_step(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]: ...


@dataclass
class RolloutEvalConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    rollout_horizon: int = 10
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 23
    export_csv: str = ""


def train_model(model: nn.Module, cfg: RolloutEvalConfig, label: str) -> None:
    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    print(f"[{label}] train steps={cfg.steps} batch={cfg.batch_size} device={device}")
    for step in range(1, cfg.steps + 1):
        batch = batcher.sample(cfg.batch_size, device)
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, cfg.steps // 5) == 0:
            print(
                f"[{label}] step={step:04d} total={float(metrics['total'].item()):.4f} "
                f"obs={float(metrics['obs_mse'].item()):.4f} reward={float(metrics['reward_mse'].item()):.4f}"
            )


@torch.no_grad()
def evaluate_true_rollout(model: WorldModelLike, cfg: RolloutEvalConfig, label: str) -> List[Dict[str, float]]:
    device = torch.device(cfg.device)
    rows: List[Dict[str, float]] = []
    for env_name, env in rollout_envs(max_obs_dim=cfg.max_obs_dim).items():
        obs_mse_by_h = torch.zeros(cfg.rollout_horizon, device=device)
        reward_mse_by_h = torch.zeros(cfg.rollout_horizon, device=device)
        done_bce_by_h = torch.zeros(cfg.rollout_horizon, device=device)
        count = 0
        for _ in range(cfg.eval_batches):
            true_obs = env.reset(cfg.batch_size, device)
            pred_obs = true_obs.clone()
            for h in range(cfg.rollout_horizon):
                actions = env.sample_actions(cfg.batch_size, device)
                true_step = env.step_from_obs(true_obs, actions)
                pred = model.predict_step(pred_obs, actions)
                pred_next_obs = pred["next_obs"]
                pred_reward = pred["reward"]
                pred_done_prob = pred["done_prob"].clamp(1e-5, 1.0 - 1e-5)

                obs_mse_by_h[h] += F.mse_loss(pred_next_obs[:, : env.obs_dim], true_step.next_obs[:, : env.obs_dim])
                reward_mse_by_h[h] += F.mse_loss(pred_reward, true_step.rewards)
                done_bce_by_h[h] += F.binary_cross_entropy(pred_done_prob, true_step.dones)

                true_obs = true_step.next_obs
                pred_obs = pred_next_obs.detach()
                count += 1

        denom = max(1, cfg.eval_batches)
        obs_mse_by_h = obs_mse_by_h / denom
        reward_mse_by_h = reward_mse_by_h / denom
        done_bce_by_h = done_bce_by_h / denom
        row = {
            "model": label,
            "env": env_name,
            "horizon": float(cfg.rollout_horizon),
            "rollout_obs_mse_h1": float(obs_mse_by_h[0].item()),
            "rollout_obs_mse_hN": float(obs_mse_by_h[-1].item()),
            "rollout_obs_mse_avg": float(obs_mse_by_h.mean().item()),
            "rollout_reward_mse_avg": float(reward_mse_by_h.mean().item()),
            "rollout_done_bce_avg": float(done_bce_by_h.mean().item()),
        }
        rows.append(row)
        print(
            f"[{label}:{env_name}] h1_obs={row['rollout_obs_mse_h1']:.4f} "
            f"h{cfg.rollout_horizon}_obs={row['rollout_obs_mse_hN']:.4f} "
            f"avg_obs={row['rollout_obs_mse_avg']:.4f} reward={row['rollout_reward_mse_avg']:.4f}"
        )
    return rows


def run(cfg: RolloutEvalConfig) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    sef = UniversalWorldModel(
        WorldModelConfig(
            max_obs_dim=cfg.max_obs_dim,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.num_actions,
        )
    ).to(device)
    mlp = MLPWorldModel(
        MLPWorldModelConfig(
            max_obs_dim=cfg.max_obs_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.num_actions,
        )
    ).to(device)

    train_model(sef, cfg, "sef_gram")
    train_model(mlp, cfg, "mlp_baseline")

    rows: List[Dict[str, float]] = []
    rows.extend(evaluate_true_rollout(sef, cfg, "sef_gram"))
    rows.extend(evaluate_true_rollout(mlp, cfg, "mlp_baseline"))

    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> RolloutEvalConfig:
    parser = argparse.ArgumentParser(description="True multi-step rollout evaluation for SEF-GRAM world model")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--export-csv", type=str, default="")
    return RolloutEvalConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
