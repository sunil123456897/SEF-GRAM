from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_model import UniversalWorldModel, WorldBatch, WorldModelConfig, pad_obs


@dataclass
class MemoryKeyDoorConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    rollout_horizon: int = 10
    size: int = 6
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 37
    export_csv: str = ""


class MemoryKeyDoorSequenceEnv:
    """Partially observable key-door sequence environment.

    At t=0 the observation includes full key/door/goal facts. At later steps,
    key/door/goal are masked to zero and only agent position + has_key remain.
    Predicting blocked movement, key pickup, reward, and done therefore requires
    carrying the earlier facts in recurrent memory.
    """

    num_actions = 4
    obs_dim = 9

    def __init__(self, size: int = 6, max_obs_dim: int = 16):
        self.size = size
        self.max_obs_dim = max_obs_dim

    def _normalize_state(self, state: Dict[str, torch.Tensor], reveal_facts: bool) -> torch.Tensor:
        n = self.size
        scale = float(n - 1)
        batch_size = state["agent"].shape[0]
        if reveal_facts:
            facts = torch.cat([state["key"], state["door"], state["goal"]], dim=-1).float() / scale
        else:
            facts = torch.zeros(batch_size, 6, device=state["agent"].device)
        obs = torch.cat([state["agent"].float() / scale, facts, state["has_key"].float().unsqueeze(-1)], dim=-1)
        return pad_obs(obs, self.max_obs_dim)

    def reset_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        n = self.size
        return {
            "agent": torch.randint(0, n, (batch_size, 2), device=device),
            "key": torch.randint(0, n, (batch_size, 2), device=device),
            "door": torch.randint(0, n, (batch_size, 2), device=device),
            "goal": torch.randint(0, n, (batch_size, 2), device=device),
            "has_key": torch.randint(0, 2, (batch_size,), device=device),
        }

    def initial_obs(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._normalize_state(state, reveal_facts=True)

    def hidden_obs(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._normalize_state(state, reveal_facts=False)

    def sample_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_actions, (batch_size,), device=device)

    def step_state(self, state: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        n = self.size
        agent = state["agent"]
        ax, ay = agent[:, 0], agent[:, 1]
        nx = ax.clone()
        ny = ay.clone()
        actions = actions.long().view(-1)
        ny = torch.where(actions == 0, (ny - 1).clamp(0, n - 1), ny)
        ny = torch.where(actions == 1, (ny + 1).clamp(0, n - 1), ny)
        nx = torch.where(actions == 2, (nx - 1).clamp(0, n - 1), nx)
        nx = torch.where(actions == 3, (nx + 1).clamp(0, n - 1), nx)

        kx, ky = state["key"][:, 0], state["key"][:, 1]
        dx, dy = state["door"][:, 0], state["door"][:, 1]
        gx, gy = state["goal"][:, 0], state["goal"][:, 1]
        has_key = state["has_key"]

        blocked = (nx == dx) & (ny == dy) & (has_key == 0)
        nx = torch.where(blocked, ax, nx)
        ny = torch.where(blocked, ay, ny)

        collected_key = (nx == kx) & (ny == ky)
        next_has_key = ((has_key == 1) | collected_key).long()
        done = ((next_has_key == 1) & (nx == gx) & (ny == gy)).float()
        reward = torch.full_like(done, -0.02)
        reward = torch.where(blocked, torch.full_like(reward, -0.1), reward)
        reward = torch.where((has_key == 0) & (next_has_key == 1), torch.full_like(reward, 0.2), reward)
        reward = torch.where(done > 0, torch.ones_like(reward), reward)

        next_state = {
            "agent": torch.stack([nx, ny], dim=-1),
            "key": state["key"],
            "door": state["door"],
            "goal": state["goal"],
            "has_key": next_has_key,
        }
        return next_state, reward, done

    def sample_sequence(self, batch_size: int, horizon: int, device: torch.device):
        state = self.reset_state(batch_size, device)
        full_initial_obs = self.initial_obs(state)
        current_obs = full_initial_obs
        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []
        for _ in range(horizon):
            action = self.sample_actions(batch_size, device)
            next_state, reward, done = self.step_state(state, action)
            next_obs = self.hidden_obs(next_state)
            observations.append(current_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            state = next_state
            current_obs = next_obs
        return {
            "initial_obs": full_initial_obs,
            "obs": torch.stack(observations, dim=1),
            "actions": torch.stack(actions, dim=1),
            "next_obs": torch.stack(next_observations, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "dones": torch.stack(dones, dim=1),
        }


def sequence_loss_sef(model: UniversalWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    state = model.init_state(seq["initial_obs"], sample_context=True)
    obs_losses = []
    reward_losses = []
    done_losses = []
    for t in range(cfg.rollout_horizon):
        state, pred = model.stateful_step(state, seq["actions"][:, t])
        obs_losses.append(F.mse_loss(pred["next_obs"][:, : MemoryKeyDoorSequenceEnv.obs_dim], seq["next_obs"][:, t, : MemoryKeyDoorSequenceEnv.obs_dim]))
        reward_losses.append(F.mse_loss(pred["reward"], seq["rewards"][:, t]))
        done_losses.append(F.binary_cross_entropy_with_logits(pred["done_logit"], seq["dones"][:, t]))
    obs_loss = torch.stack(obs_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()
    done_loss = torch.stack(done_losses).mean()
    total = obs_loss + reward_loss + 0.2 * done_loss
    return total, {"obs_mse": obs_loss.detach(), "reward_mse": reward_loss.detach(), "done_bce": done_loss.detach()}


def sequence_loss_mlp(model: MLPWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    obs_losses = []
    reward_losses = []
    done_losses = []
    pred_obs = seq["initial_obs"]
    for t in range(cfg.rollout_horizon):
        dummy = WorldBatch(
            obs=pred_obs,
            actions=seq["actions"][:, t],
            next_obs=torch.zeros(pred_obs.shape[0], cfg.max_obs_dim, device=pred_obs.device),
            rewards=torch.zeros(pred_obs.shape[0], device=pred_obs.device),
            dones=torch.zeros(pred_obs.shape[0], device=pred_obs.device),
        )
        pred = model.forward(dummy)
        pred_next_obs = pred["pred_next_obs"]
        obs_losses.append(F.mse_loss(pred_next_obs[:, : MemoryKeyDoorSequenceEnv.obs_dim], seq["next_obs"][:, t, : MemoryKeyDoorSequenceEnv.obs_dim]))
        reward_losses.append(F.mse_loss(pred["pred_reward"], seq["rewards"][:, t]))
        done_losses.append(F.binary_cross_entropy_with_logits(pred["pred_done_logit"], seq["dones"][:, t]))
        pred_obs = pred_next_obs
    obs_loss = torch.stack(obs_losses).mean()
    reward_loss = torch.stack(reward_losses).mean()
    done_loss = torch.stack(done_losses).mean()
    total = obs_loss + reward_loss + 0.2 * done_loss
    return total, {"obs_mse": obs_loss.detach(), "reward_mse": reward_loss.detach(), "done_bce": done_loss.detach()}


@torch.no_grad()
def evaluate_memory_keydoor(model, label: str, cfg: MemoryKeyDoorConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    obs_values = []
    reward_values = []
    done_values = []
    for _ in range(cfg.eval_batches):
        seq = env.sample_sequence(cfg.batch_size, cfg.rollout_horizon, device)
        if label.startswith("sef"):
            loss, metrics = sequence_loss_sef(model, seq, cfg)
        else:
            loss, metrics = sequence_loss_mlp(model, seq, cfg)
        obs_values.append(float(metrics["obs_mse"].item()))
        reward_values.append(float(metrics["reward_mse"].item()))
        done_values.append(float(metrics["done_bce"].item()))
    return {
        "model": label,
        "env": "memory_key_door",
        "horizon": float(cfg.rollout_horizon),
        "rollout_obs_mse_avg": sum(obs_values) / len(obs_values),
        "rollout_reward_mse_avg": sum(reward_values) / len(reward_values),
        "rollout_done_bce_avg": sum(done_values) / len(done_values),
    }


def train_memory_keydoor(cfg: MemoryKeyDoorConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    sef = UniversalWorldModel(
        WorldModelConfig(max_obs_dim=cfg.max_obs_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions)
    ).to(device)
    mlp = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=cfg.max_obs_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions)).to(device)
    optimizers = {
        "sef_gram_memory": torch.optim.AdamW(sef.parameters(), lr=cfg.lr, weight_decay=1e-4),
        "mlp_memory": torch.optim.AdamW(mlp.parameters(), lr=cfg.lr, weight_decay=1e-4),
    }
    models = {"sef_gram_memory": sef, "mlp_memory": mlp}

    for label, model in models.items():
        print(f"[{label}] train steps={cfg.steps} batch={cfg.batch_size} horizon={cfg.rollout_horizon} device={device}")
        optimizer = optimizers[label]
        for step in range(1, cfg.steps + 1):
            seq = env.sample_sequence(cfg.batch_size, cfg.rollout_horizon, device)
            model.train()
            optimizer.zero_grad()
            if label.startswith("sef"):
                loss, metrics = sequence_loss_sef(model, seq, cfg)
            else:
                loss, metrics = sequence_loss_mlp(model, seq, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % max(1, cfg.steps // 5) == 0:
                print(
                    f"[{label}] step={step:04d} total={float(loss.item()):.4f} "
                    f"obs={float(metrics['obs_mse'].item()):.4f} "
                    f"reward={float(metrics['reward_mse'].item()):.4f} "
                    f"done={float(metrics['done_bce'].item()):.4f}"
                )
    return models


def run(cfg: MemoryKeyDoorConfig) -> List[Dict[str, float]]:
    models = train_memory_keydoor(cfg)
    rows = [evaluate_memory_keydoor(model, label, cfg) for label, model in models.items()]
    print("\n[memory key-door summary]")
    for row in rows:
        print(
            f"{row['model']} obs={row['rollout_obs_mse_avg']:.4f} "
            f"reward={row['rollout_reward_mse_avg']:.4f} done={row['rollout_done_bce_avg']:.4f}"
        )
    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> MemoryKeyDoorConfig:
    parser = argparse.ArgumentParser(description="Partially observable key-door memory benchmark")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--rollout-horizon", type=int, default=10)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--export-csv", type=str, default="")
    return MemoryKeyDoorConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
