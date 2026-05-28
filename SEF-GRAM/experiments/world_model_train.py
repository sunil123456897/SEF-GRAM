from __future__ import annotations

from pathlib import Path
import argparse
import sys
from dataclasses import dataclass
from typing import Dict

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.world_envs import build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig


@dataclass
class TrainConfig:
    steps: int = 500
    batch_size: int = 128
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 13


def train(cfg: TrainConfig) -> Dict[str, float]:
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(f"UniversalWorldModel training | steps={cfg.steps} batch={cfg.batch_size} device={device}")
    last_metrics: Dict[str, float] = {}
    for step in range(1, cfg.steps + 1):
        batch = batcher.sample(cfg.batch_size, device)
        model.train()
        optimizer.zero_grad()
        loss, metrics = model.loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        last_metrics = {key: float(value.item()) for key, value in metrics.items()}
        if step == 1 or step % max(1, cfg.steps // 10) == 0:
            print(
                f"step={step:04d} total={last_metrics['total']:.4f} "
                f"latent_mse={last_metrics['latent_mse']:.4f} "
                f"obs={last_metrics['obs_mse']:.4f} "
                f"reward={last_metrics['reward_mse']:.4f} "
                f"done={last_metrics['done_bce']:.4f}"
            )
    return last_metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SEF-GRAM Phase 2 universal latent world model")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    return TrainConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    train(parse_args())
