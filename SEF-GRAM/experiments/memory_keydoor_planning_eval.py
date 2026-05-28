from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.memory_keydoor_compare import MemoryKeyDoorConfig, MemoryKeyDoorSequenceEnv, train_memory_keydoor
from sef_gram.world_baselines import GRUWorldModel, MLPWorldModel
from sef_gram.world_model import UniversalWorldModel, WorldBatch


@dataclass
class PlanningConfig:
    steps: int = 500
    batch_size: int = 128
    eval_batches: int = 20
    plan_horizon: int = 12
    num_candidates: int = 128
    size: int = 6
    max_obs_dim: int = 16
    latent_dim: int = 64
    hidden_dim: int = 128
    num_actions: int = 4
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 43
    done_bonus: float = 3.0
    key_bonus: float = 0.5
    force_no_key_start: bool = True
    export_csv: str = ""


def _path_actions(start_xy: torch.Tensor, target_xy: torch.Tensor, horizon: int, device: torch.device) -> torch.Tensor:
    actions: List[int] = []
    x, y = int(start_xy[0].item()), int(start_xy[1].item())
    tx, ty = int(target_xy[0].item()), int(target_xy[1].item())
    while x < tx and len(actions) < horizon:
        actions.append(3)
        x += 1
    while x > tx and len(actions) < horizon:
        actions.append(2)
        x -= 1
    while y < ty and len(actions) < horizon:
        actions.append(1)
        y += 1
    while y > ty and len(actions) < horizon:
        actions.append(0)
        y -= 1
    if len(actions) < horizon:
        actions.extend([1] * (horizon - len(actions)))
    return torch.tensor(actions[:horizon], dtype=torch.long, device=device)


def _decode_initial_positions(initial_obs: torch.Tensor, size: int) -> torch.Tensor:
    scale = float(size - 1)
    return torch.round(initial_obs[:, :8].clamp(0.0, 1.0) * scale).long().clamp(0, size - 1)


def build_candidate_actions(initial_obs: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    """Build action candidates from random shooting plus simple visible-fact paths.

    The environment exposes key/door/goal at t=0, so the planner is allowed to use
    these facts to propose candidates. The learned model still scores candidates.
    """

    device = initial_obs.device
    batch_size = initial_obs.shape[0]
    candidates = torch.randint(0, cfg.num_actions, (batch_size, cfg.num_candidates, cfg.plan_horizon), device=device)
    pos = _decode_initial_positions(initial_obs, cfg.size)

    for b in range(batch_size):
        agent = pos[b, 0:2]
        key = pos[b, 2:4]
        goal = pos[b, 6:8]
        if cfg.num_candidates >= 1:
            to_key = _path_actions(agent, key, cfg.plan_horizon, device)
            key_pos_after = key
            remaining = max(0, cfg.plan_horizon - int((to_key != 1).numel()))
            key_then_goal = torch.cat(
                [
                    _path_actions(agent, key, cfg.plan_horizon, device),
                    _path_actions(key_pos_after, goal, cfg.plan_horizon, device),
                ]
            )[: cfg.plan_horizon]
            candidates[b, 0] = key_then_goal
        if cfg.num_candidates >= 2:
            candidates[b, 1] = _path_actions(agent, goal, cfg.plan_horizon, device)
        if cfg.num_candidates >= 3:
            path = _path_actions(agent, key, cfg.plan_horizon, device)
            cut = min(cfg.plan_horizon, max(1, int((agent - key).abs().sum().item())))
            mixed = torch.randint(0, cfg.num_actions, (cfg.plan_horizon,), device=device)
            mixed[:cut] = path[:cut]
            candidates[b, 2] = mixed
    return candidates


def score_candidates_sef(model: UniversalWorldModel, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    batch_size, num_candidates, horizon = candidates.shape
    flat_obs = initial_obs[:, None, :].expand(batch_size, num_candidates, cfg.max_obs_dim).reshape(batch_size * num_candidates, cfg.max_obs_dim)
    state = model.init_state(flat_obs, sample_context=False)
    score = torch.zeros(batch_size * num_candidates, device=initial_obs.device)
    max_done = torch.zeros_like(score)
    max_key = torch.zeros_like(score)
    flat_actions = candidates.reshape(batch_size * num_candidates, horizon)
    for t in range(horizon):
        state, pred = model.stateful_step(state, flat_actions[:, t])
        done_prob = torch.sigmoid(pred["done_logit"])
        key_prob = pred["next_obs"][:, 8].clamp(0.0, 1.0)
        score = score + pred["reward"]
        max_done = torch.maximum(max_done, done_prob)
        max_key = torch.maximum(max_key, key_prob)
    score = score + cfg.done_bonus * max_done + cfg.key_bonus * max_key
    return score.view(batch_size, num_candidates)


def score_candidates_gru(model: GRUWorldModel, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    batch_size, num_candidates, horizon = candidates.shape
    current_obs = initial_obs[:, None, :].expand(batch_size, num_candidates, cfg.max_obs_dim).reshape(batch_size * num_candidates, cfg.max_obs_dim)
    hidden = model.initial_state(batch_size * num_candidates, initial_obs.device)
    score = torch.zeros(batch_size * num_candidates, device=initial_obs.device)
    max_done = torch.zeros_like(score)
    max_key = torch.zeros_like(score)
    flat_actions = candidates.reshape(batch_size * num_candidates, horizon)
    for t in range(horizon):
        hidden, pred = model.step(current_obs, flat_actions[:, t], hidden)
        current_obs = pred["pred_next_obs"]
        done_prob = torch.sigmoid(pred["pred_done_logit"])
        key_prob = current_obs[:, 8].clamp(0.0, 1.0)
        score = score + pred["pred_reward"]
        max_done = torch.maximum(max_done, done_prob)
        max_key = torch.maximum(max_key, key_prob)
    score = score + cfg.done_bonus * max_done + cfg.key_bonus * max_key
    return score.view(batch_size, num_candidates)


def score_candidates_mlp(model: MLPWorldModel, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    batch_size, num_candidates, horizon = candidates.shape
    current_obs = initial_obs[:, None, :].expand(batch_size, num_candidates, cfg.max_obs_dim).reshape(batch_size * num_candidates, cfg.max_obs_dim)
    score = torch.zeros(batch_size * num_candidates, device=initial_obs.device)
    max_done = torch.zeros_like(score)
    max_key = torch.zeros_like(score)
    flat_actions = candidates.reshape(batch_size * num_candidates, horizon)
    for t in range(horizon):
        dummy = WorldBatch(
            obs=current_obs,
            actions=flat_actions[:, t],
            next_obs=torch.zeros_like(current_obs),
            rewards=torch.zeros(current_obs.shape[0], device=current_obs.device),
            dones=torch.zeros(current_obs.shape[0], device=current_obs.device),
        )
        pred = model.forward(dummy)
        current_obs = pred["pred_next_obs"]
        done_prob = torch.sigmoid(pred["pred_done_logit"])
        key_prob = current_obs[:, 8].clamp(0.0, 1.0)
        score = score + pred["pred_reward"]
        max_done = torch.maximum(max_done, done_prob)
        max_key = torch.maximum(max_key, key_prob)
    score = score + cfg.done_bonus * max_done + cfg.key_bonus * max_key
    return score.view(batch_size, num_candidates)


def score_candidates(model, label: str, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    if label.startswith("sef"):
        return score_candidates_sef(model, initial_obs, candidates, cfg)
    if label.startswith("gru"):
        return score_candidates_gru(model, initial_obs, candidates, cfg)
    if label.startswith("mlp"):
        return score_candidates_mlp(model, initial_obs, candidates, cfg)
    raise ValueError(f"unknown model label: {label}")


def execute_action_sequences(env: MemoryKeyDoorSequenceEnv, state: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, torch.Tensor]:
    total_reward = torch.zeros(actions.shape[0], device=actions.device)
    ever_done = torch.zeros(actions.shape[0], device=actions.device)
    ever_key = torch.zeros(actions.shape[0], device=actions.device)
    blocked_count = torch.zeros(actions.shape[0], device=actions.device)
    current_state = {key: value.clone() for key, value in state.items()}
    for t in range(actions.shape[1]):
        current_state, reward, done, events = env.step_state(current_state, actions[:, t])
        total_reward = total_reward + reward
        ever_done = torch.maximum(ever_done, done)
        ever_key = torch.maximum(ever_key, current_state["has_key"].float())
        blocked_count = blocked_count + events["blocked"]
    return {
        "total_reward": total_reward,
        "success": ever_done,
        "has_key": ever_key,
        "blocked_count": blocked_count,
    }


@torch.no_grad()
def evaluate_planning(models: Dict[str, object], cfg: PlanningConfig) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed + 1000)
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    accum: Dict[str, Dict[str, List[float]]] = {}

    labels = ["random_policy"] + list(models.keys())
    for label in labels:
        accum[label] = {"success_rate": [], "key_rate": [], "avg_total_reward": [], "avg_blocked_count": []}

    for _ in range(cfg.eval_batches):
        state = env.reset_state(cfg.batch_size, device)
        if cfg.force_no_key_start:
            state["has_key"] = torch.zeros_like(state["has_key"])
        initial_obs = env.initial_obs(state)
        candidates = build_candidate_actions(initial_obs, cfg)

        random_actions = torch.randint(0, cfg.num_actions, (cfg.batch_size, cfg.plan_horizon), device=device)
        random_result = execute_action_sequences(env, state, random_actions)
        accum["random_policy"]["success_rate"].append(float(random_result["success"].mean().item()))
        accum["random_policy"]["key_rate"].append(float(random_result["has_key"].mean().item()))
        accum["random_policy"]["avg_total_reward"].append(float(random_result["total_reward"].mean().item()))
        accum["random_policy"]["avg_blocked_count"].append(float(random_result["blocked_count"].mean().item()))

        for label, model in models.items():
            scores = score_candidates(model, label, initial_obs, candidates, cfg)
            best_idx = scores.argmax(dim=1)
            best_actions = candidates[torch.arange(cfg.batch_size, device=device), best_idx]
            result = execute_action_sequences(env, state, best_actions)
            accum[label]["success_rate"].append(float(result["success"].mean().item()))
            accum[label]["key_rate"].append(float(result["has_key"].mean().item()))
            accum[label]["avg_total_reward"].append(float(result["total_reward"].mean().item()))
            accum[label]["avg_blocked_count"].append(float(result["blocked_count"].mean().item()))

    rows: List[Dict[str, float]] = []
    for label, metrics in accum.items():
        row: Dict[str, float] = {"model": label, "env": "memory_key_door_planning"}
        for key, values in metrics.items():
            row[key] = sum(values) / len(values)
        rows.append(row)
        print(
            f"[{label}] success={row['success_rate']:.2%} key={row['key_rate']:.2%} "
            f"reward={row['avg_total_reward']:.4f} blocked={row['avg_blocked_count']:.4f}"
        )
    return rows


def train_models(cfg: PlanningConfig) -> Dict[str, object]:
    memory_cfg = MemoryKeyDoorConfig(
        steps=cfg.steps,
        batch_size=cfg.batch_size,
        eval_batches=1,
        rollout_horizon=cfg.plan_horizon,
        size=cfg.size,
        max_obs_dim=cfg.max_obs_dim,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
        num_actions=cfg.num_actions,
        lr=cfg.lr,
        device=cfg.device,
        seed=cfg.seed,
    )
    return train_memory_keydoor(memory_cfg)


def run(cfg: PlanningConfig) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed)
    models = train_models(cfg)
    rows = evaluate_planning(models, cfg)
    if cfg.export_csv:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")
    return rows


def parse_args() -> PlanningConfig:
    parser = argparse.ArgumentParser(description="Model-based planning evaluation for partially observable key-door")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--plan-horizon", type=int, default=12)
    parser.add_argument("--num-candidates", type=int, default=128)
    parser.add_argument("--size", type=int, default=6)
    parser.add_argument("--max-obs-dim", type=int, default=16)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--done-bonus", type=float, default=3.0)
    parser.add_argument("--key-bonus", type=float, default=0.5)
    parser.add_argument("--allow-key-start", action="store_true")
    parser.add_argument("--export-csv", type=str, default="")
    args = parser.parse_args()
    return PlanningConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        plan_horizon=args.plan_horizon,
        num_candidates=args.num_candidates,
        size=args.size,
        max_obs_dim=args.max_obs_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        done_bonus=args.done_bonus,
        key_bonus=args.key_bonus,
        force_no_key_start=not args.allow_key_start,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    run(parse_args())
