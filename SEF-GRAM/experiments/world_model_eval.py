from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Protocol

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_envs import build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldBatch, WorldModelConfig


class WorldModelLike(Protocol):
    def loss(self, batch: WorldBatch): ...
    def predict_step(self, obs: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]: ...


@dataclass
class EvalConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    rollout_horizon: int = 5
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 17
    export_csv: str = ""


def train_model(model: nn.Module, cfg: EvalConfig, label: str) -> None:
    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    if isinstance(model, UniversalWorldModel):
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
            total = float(metrics["total"].item())
            obs = float(metrics["obs_mse"].item())
            reward = float(metrics["reward_mse"].item())
            done = float(metrics["done_bce"].item())
            de = float(metrics.get("dirichlet_energy", -1.0))
            print(f"[{label}] step={step:04d} total={total:.4f} obs={obs:.4f} reward={reward:.4f} done={done:.4f} E_Dirichlet={de:.4f}")


@torch.no_grad()
def evaluate_one_step(model: WorldModelLike, cfg: EvalConfig, label: str) -> Dict[str, float]:
    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    totals: List[float] = []
    obs_values: List[float] = []
    reward_values: List[float] = []
    done_values: List[float] = []
    for _ in range(cfg.eval_batches):
        batch = batcher.sample(cfg.batch_size, device)
        _, metrics = model.loss(batch)
        totals.append(float(metrics["total"].item()))
        obs_values.append(float(metrics["obs_mse"].item()))
        reward_values.append(float(metrics["reward_mse"].item()))
        done_values.append(float(metrics["done_bce"].item()))
    return {
        "model": label,
        "one_step_total": sum(totals) / len(totals),
        "one_step_obs_mse": sum(obs_values) / len(obs_values),
        "one_step_reward_mse": sum(reward_values) / len(reward_values),
        "one_step_done_bce": sum(done_values) / len(done_values),
    }


@torch.no_grad()
def evaluate_open_loop_proxy(model: WorldModelLike, cfg: EvalConfig, label: str) -> Dict[str, float]:
    """Open-loop proxy on independently sampled transition batches.

    The toy generators are stateless samplers, not full simulators, so this is not a
    true environment rollout. It still measures whether repeated model predictions
    stay numerically stable when fed back as observations.
    """

    device = torch.device(cfg.device)
    batcher = build_default_mixed_batcher(max_obs_dim=cfg.max_obs_dim)
    obs_errors: List[float] = []
    reward_abs: List[float] = []
    done_mean: List[float] = []
    batch = batcher.sample(cfg.batch_size, device)
    obs = batch.obs
    for _ in range(cfg.rollout_horizon):
        actions = torch.randint(0, cfg.num_actions, (cfg.batch_size,), device=device)
        pred = model.predict_step(obs, actions)
        obs = pred["next_obs"]
        obs_errors.append(float((obs.pow(2).mean()).item()))
        reward_abs.append(float(pred["reward"].abs().mean().item()))
        done_mean.append(float(pred["done_prob"].mean().item()))
    return {
        "model": label,
        "open_loop_obs_energy": sum(obs_errors) / len(obs_errors),
        "open_loop_abs_reward": sum(reward_abs) / len(reward_abs),
        "open_loop_done_prob": sum(done_mean) / len(done_mean),
    }


def merge_rows(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    out = dict(a)
    out.update({key: value for key, value in b.items() if key != "model"})
    return out


def run(cfg: EvalConfig) -> List[Dict[str, float]]:
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

    rows = []
    for label, model in (("sef_gram", sef), ("mlp_baseline", mlp)):
        one = evaluate_one_step(model, cfg, label)
        rollout = evaluate_open_loop_proxy(model, cfg, label)
        row = merge_rows(one, rollout)
        rows.append(row)
        print(
            f"[{label}] eval total={row['one_step_total']:.4f} obs={row['one_step_obs_mse']:.4f} "
            f"reward={row['one_step_reward_mse']:.4f} open_loop_energy={row['open_loop_obs_energy']:.4f}"
        )

    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate SEF-GRAM world model against MLP baseline")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=5)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--export-csv", type=str, default="")
    return EvalConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
