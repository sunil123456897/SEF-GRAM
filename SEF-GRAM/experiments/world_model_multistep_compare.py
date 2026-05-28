from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.world_model_rollout_eval import evaluate_true_rollout
from experiments.world_model_multistep_train import multi_env_rollout_loss
from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_envs import build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig


@dataclass
class MultiStepCompareConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    rollout_horizon: int = 10
    rollout_loss_weight: float = 1.0
    one_step_loss_weight: float = 1.0
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 31
    export_csv: str = ""


def build_models(cfg: MultiStepCompareConfig, device: torch.device) -> List[Tuple[str, nn.Module]]:
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
    return [("sef_gram_multistep", sef), ("mlp_multistep", mlp)]


def train_with_multistep_loss(model: nn.Module, cfg: MultiStepCompareConfig, label: str) -> None:
    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    if label.startswith("sef"):
        optimizer = MuonWithAuxAdam(
            model.parameters(),
            lr=0.02,
            momentum=0.95,
            ns_steps=5,
            adamw_lr=cfg.lr,
            adamw_betas=(0.9, 0.95),
            adamw_wd=1e-4,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(
        f"[{label}] train steps={cfg.steps} batch={cfg.batch_size} "
        f"horizon={cfg.rollout_horizon} device={device}"
    )
    for step in range(1, cfg.steps + 1):
        batch = batcher.sample(cfg.batch_size, device)
        model.train()
        optimizer.zero_grad()
        one_step_loss, one_step_metrics = model.loss(batch)
        rollout_loss, rollout_metrics = multi_env_rollout_loss(model, cfg, device)
        total = cfg.one_step_loss_weight * one_step_loss + cfg.rollout_loss_weight * rollout_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.steps // 5) == 0:
            de = one_step_metrics.get("dirichlet_energy", torch.tensor(-1.0))
            print(
                f"[{label}] step={step:04d} total={float(total.item()):.4f} "
                f"one={float(one_step_metrics['total'].item()):.4f} "
                f"rollout={float(rollout_metrics['rollout_total'].item()):.4f} "
                f"roll_obs={float(rollout_metrics['rollout_obs_mse'].item()):.4f} "
                f"roll_reward={float(rollout_metrics['rollout_reward_mse'].item()):.4f} "
                f"E_Dirichlet={float(de.item()):.4f}"
            )


def run(cfg: MultiStepCompareConfig) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    rows: List[Dict[str, float]] = []

    for label, model in build_models(cfg, device):
        train_with_multistep_loss(model, cfg, label)
        rows.extend(evaluate_true_rollout(model, cfg, label))

    print("\n[multistep comparison summary]")
    for row in rows:
        print(
            f"{row['model']}:{row['env']} "
            f"h1={row['rollout_obs_mse_h1']:.4f} "
            f"hN={row['rollout_obs_mse_hN']:.4f} "
            f"avg={row['rollout_obs_mse_avg']:.4f} "
            f"reward={row['rollout_reward_mse_avg']:.4f}"
        )

    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> MultiStepCompareConfig:
    parser = argparse.ArgumentParser(description="Fair multi-step rollout training comparison: SEF-GRAM vs MLP baseline")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0)
    parser.add_argument("--one-step-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--export-csv", type=str, default="")
    return MultiStepCompareConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
