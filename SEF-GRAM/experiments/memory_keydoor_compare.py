from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.world_baselines import GRUWorldModel, GRUWorldModelConfig, MLPWorldModel, MLPWorldModelConfig
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

    def step_state(
        self,
        state: Dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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

        naive_moved = (nx != ax) | (ny != ay)
        blocked = (nx == dx) & (ny == dy) & (has_key == 0)
        nx = torch.where(blocked, ax, nx)
        ny = torch.where(blocked, ay, ny)

        collected_key = (nx == kx) & (ny == ky)
        key_pickup = (has_key == 0) & collected_key
        next_has_key = ((has_key == 1) | collected_key).long()
        done = ((next_has_key == 1) & (nx == gx) & (ny == gy)).float()
        reward = torch.full_like(done, -0.02)
        reward = torch.where(blocked, torch.full_like(reward, -0.1), reward)
        reward = torch.where(key_pickup, torch.full_like(reward, 0.2), reward)
        reward = torch.where(done > 0, torch.ones_like(reward), reward)

        next_state = {
            "agent": torch.stack([nx, ny], dim=-1),
            "key": state["key"],
            "door": state["door"],
            "goal": state["goal"],
            "has_key": next_has_key,
        }
        events = {"blocked": blocked.float(), "key_pickup": key_pickup.float(), "naive_moved": naive_moved.float()}
        return next_state, reward, done, events

    def sample_sequence(self, batch_size: int, horizon: int, device: torch.device):
        state = self.reset_state(batch_size, device)
        full_initial_obs = self.initial_obs(state)
        current_obs = full_initial_obs
        observations = []
        actions = []
        next_observations = []
        rewards = []
        dones = []
        blocked_events = []
        key_pickup_events = []
        naive_moved_events = []
        for _ in range(horizon):
            action = self.sample_actions(batch_size, device)
            next_state, reward, done, events = self.step_state(state, action)
            next_obs = self.hidden_obs(next_state)
            observations.append(current_obs)
            actions.append(action)
            next_observations.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            blocked_events.append(events["blocked"])
            key_pickup_events.append(events["key_pickup"])
            naive_moved_events.append(events["naive_moved"])
            state = next_state
            current_obs = next_obs
        return {
            "initial_obs": full_initial_obs,
            "obs": torch.stack(observations, dim=1),
            "actions": torch.stack(actions, dim=1),
            "next_obs": torch.stack(next_observations, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "dones": torch.stack(dones, dim=1),
            "blocked": torch.stack(blocked_events, dim=1),
            "key_pickup": torch.stack(key_pickup_events, dim=1),
            "naive_moved": torch.stack(naive_moved_events, dim=1),
        }


def rollout_predict_sef(model: UniversalWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Dict[str, torch.Tensor]:
    state = model.init_state(seq["initial_obs"], sample_context=True)
    pred_obs = []
    pred_rewards = []
    pred_done_logits = []
    for t in range(cfg.rollout_horizon):
        state, pred = model.stateful_step(state, seq["actions"][:, t])
        pred_obs.append(pred["next_obs"])
        pred_rewards.append(pred["reward"])
        pred_done_logits.append(pred["done_logit"])
    return {"next_obs": torch.stack(pred_obs, dim=1), "reward": torch.stack(pred_rewards, dim=1), "done_logit": torch.stack(pred_done_logits, dim=1)}


def rollout_predict_mlp(model: MLPWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Dict[str, torch.Tensor]:
    pred_obs = []
    pred_rewards = []
    pred_done_logits = []
    current_obs = seq["initial_obs"]
    for t in range(cfg.rollout_horizon):
        dummy = WorldBatch(
            obs=current_obs,
            actions=seq["actions"][:, t],
            next_obs=torch.zeros(current_obs.shape[0], cfg.max_obs_dim, device=current_obs.device),
            rewards=torch.zeros(current_obs.shape[0], device=current_obs.device),
            dones=torch.zeros(current_obs.shape[0], device=current_obs.device),
        )
        pred = model.forward(dummy)
        current_obs = pred["pred_next_obs"]
        pred_obs.append(current_obs)
        pred_rewards.append(pred["pred_reward"])
        pred_done_logits.append(pred["pred_done_logit"])
    return {"next_obs": torch.stack(pred_obs, dim=1), "reward": torch.stack(pred_rewards, dim=1), "done_logit": torch.stack(pred_done_logits, dim=1)}


def rollout_predict_gru(model: GRUWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Dict[str, torch.Tensor]:
    pred_obs = []
    pred_rewards = []
    pred_done_logits = []
    current_obs = seq["initial_obs"]
    hidden = model.initial_state(current_obs.shape[0], current_obs.device)
    for t in range(cfg.rollout_horizon):
        hidden, pred = model.step(current_obs, seq["actions"][:, t], hidden)
        current_obs = pred["pred_next_obs"]
        pred_obs.append(current_obs)
        pred_rewards.append(pred["pred_reward"])
        pred_done_logits.append(pred["pred_done_logit"])
    return {"next_obs": torch.stack(pred_obs, dim=1), "reward": torch.stack(pred_rewards, dim=1), "done_logit": torch.stack(pred_done_logits, dim=1)}


def sequence_loss_from_predictions(pred: Dict[str, torch.Tensor], seq: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    obs_loss = F.mse_loss(pred["next_obs"][:, :, : MemoryKeyDoorSequenceEnv.obs_dim], seq["next_obs"][:, :, : MemoryKeyDoorSequenceEnv.obs_dim])
    reward_loss = F.mse_loss(pred["reward"], seq["rewards"])
    done_loss = F.binary_cross_entropy_with_logits(pred["done_logit"], seq["dones"])
    total = obs_loss + reward_loss + 0.2 * done_loss
    return total, {"obs_mse": obs_loss.detach(), "reward_mse": reward_loss.detach(), "done_bce": done_loss.detach()}


def sequence_loss_sef(model: UniversalWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return sequence_loss_from_predictions(rollout_predict_sef(model, seq, cfg), seq)


def sequence_loss_mlp(model: MLPWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return sequence_loss_from_predictions(rollout_predict_mlp(model, seq, cfg), seq)


def sequence_loss_gru(model: GRUWorldModel, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    return sequence_loss_from_predictions(rollout_predict_gru(model, seq, cfg), seq)


def predictions_for_label(model, label: str, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Dict[str, torch.Tensor]:
    if label.startswith("sef"):
        return rollout_predict_sef(model, seq, cfg)
    if label.startswith("gru"):
        return rollout_predict_gru(model, seq, cfg)
    return rollout_predict_mlp(model, seq, cfg)


def loss_for_label(model, label: str, seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if label.startswith("sef"):
        return sequence_loss_sef(model, seq, cfg)
    if label.startswith("gru"):
        return sequence_loss_gru(model, seq, cfg)
    return sequence_loss_mlp(model, seq, cfg)


def _binary_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred.float() == target.float()).float().mean().item())


def _positive_recall(pred: torch.Tensor, target: torch.Tensor) -> float:
    positives = target.float().sum()
    if positives.item() <= 0:
        return float("nan")
    return float(((pred.float() == 1.0) & (target.float() == 1.0)).float().sum().item() / positives.item())


def event_metrics_from_predictions(pred: Dict[str, torch.Tensor], seq: Dict[str, torch.Tensor], cfg: MemoryKeyDoorConfig) -> Dict[str, float]:
    scale = float(cfg.size - 1)
    pred_next_obs = pred["next_obs"]
    true_next_obs = seq["next_obs"]

    pred_has_key = (pred_next_obs[:, :, 8] > 0.5).float()
    true_has_key = (true_next_obs[:, :, 8] > 0.5).float()
    prev_true_has_key = (seq["obs"][:, :, 8] > 0.5).float()
    pred_key_pickup = ((prev_true_has_key == 0.0) & (pred_has_key == 1.0)).float()
    true_key_pickup = seq["key_pickup"].float()

    pred_done = (torch.sigmoid(pred["done_logit"]) > 0.5).float()
    true_done = seq["dones"].float()

    pred_prev_agent = torch.cat([seq["initial_obs"][:, None, :2], pred_next_obs[:, :-1, :2]], dim=1)
    pred_move_dist = (pred_next_obs[:, :, :2] - pred_prev_agent).abs().sum(dim=-1)
    pred_blocked = ((pred_move_dist < (0.5 / scale)) & (seq["naive_moved"] > 0.5)).float()
    true_blocked = seq["blocked"].float()

    return {
        "has_key_accuracy": _binary_accuracy(pred_has_key, true_has_key),
        "key_pickup_accuracy": _binary_accuracy(pred_key_pickup, true_key_pickup),
        "key_pickup_recall": _positive_recall(pred_key_pickup, true_key_pickup),
        "blocked_accuracy": _binary_accuracy(pred_blocked, true_blocked),
        "blocked_recall": _positive_recall(pred_blocked, true_blocked),
        "done_accuracy": _binary_accuracy(pred_done, true_done),
        "done_recall": _positive_recall(pred_done, true_done),
        "key_pickup_rate": float(true_key_pickup.mean().item()),
        "blocked_rate": float(true_blocked.mean().item()),
        "done_rate": float(true_done.mean().item()),
    }


@torch.no_grad()
def evaluate_memory_keydoor(model, label: str, cfg: MemoryKeyDoorConfig) -> Dict[str, float]:
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    obs_values = []
    reward_values = []
    done_values = []
    event_values: Dict[str, List[float]] = {}
    for _ in range(cfg.eval_batches):
        seq = env.sample_sequence(cfg.batch_size, cfg.rollout_horizon, device)
        pred = predictions_for_label(model, label, seq, cfg)
        _, metrics = sequence_loss_from_predictions(pred, seq)
        events = event_metrics_from_predictions(pred, seq, cfg)
        obs_values.append(float(metrics["obs_mse"].item()))
        reward_values.append(float(metrics["reward_mse"].item()))
        done_values.append(float(metrics["done_bce"].item()))
        for key, value in events.items():
            if value == value:
                event_values.setdefault(key, []).append(value)
    row = {
        "model": label,
        "env": "memory_key_door",
        "horizon": float(cfg.rollout_horizon),
        "rollout_obs_mse_avg": sum(obs_values) / len(obs_values),
        "rollout_reward_mse_avg": sum(reward_values) / len(reward_values),
        "rollout_done_bce_avg": sum(done_values) / len(done_values),
    }
    for key, values in event_values.items():
        row[key] = sum(values) / len(values) if values else float("nan")
    return row


def train_memory_keydoor(cfg: MemoryKeyDoorConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    sef = UniversalWorldModel(
        WorldModelConfig(max_obs_dim=cfg.max_obs_dim, latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions)
    ).to(device)
    mlp = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=cfg.max_obs_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions)).to(device)
    gru = GRUWorldModel(GRUWorldModelConfig(max_obs_dim=cfg.max_obs_dim, hidden_dim=cfg.hidden_dim, num_actions=cfg.num_actions)).to(device)
    models = {"sef_gram_memory": sef, "gru_memory": gru, "mlp_memory": mlp}
    optimizers = {}
    for label, model in models.items():
        if label.startswith("sef") or label.startswith("gru"):
            optimizers[label] = MuonWithAuxAdam(
                model.parameters(),
                lr=0.02,
                momentum=0.95,
                ns_steps=5,
                adamw_lr=cfg.lr,
                adamw_betas=(0.9, 0.95),
                adamw_wd=1e-4,
            )
        else:
            optimizers[label] = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    for label, model in models.items():
        print(f"[{label}] train steps={cfg.steps} batch={cfg.batch_size} horizon={cfg.rollout_horizon} device={device}")
        optimizer = optimizers[label]
        for step in range(1, cfg.steps + 1):
            seq = env.sample_sequence(cfg.batch_size, cfg.rollout_horizon, device)
            model.train()
            optimizer.zero_grad()
            loss, metrics = loss_for_label(model, label, seq, cfg)
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
            f"reward={row['rollout_reward_mse_avg']:.4f} done={row['rollout_done_bce_avg']:.4f} "
            f"has_key_acc={row.get('has_key_accuracy', float('nan')):.2%} "
            f"blocked_acc={row.get('blocked_accuracy', float('nan')):.2%} "
            f"key_pickup_acc={row.get('key_pickup_accuracy', float('nan')):.2%} "
            f"done_acc={row.get('done_accuracy', float('nan')):.2%}"
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
