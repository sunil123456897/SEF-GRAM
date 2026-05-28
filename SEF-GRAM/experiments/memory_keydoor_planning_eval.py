from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass
from typing import Dict, List

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
    key_distance_weight: float = 1.0
    goal_distance_weight: float = 2.0
    force_no_key_start: bool = True
    use_cem: bool = False
    cem_iterations: int = 4
    cem_elite_frac: float = 0.25
    export_csv: str = ""


def _path_action_list(start_xy: torch.Tensor, target_xy: torch.Tensor) -> List[int]:
    """Shortest Manhattan action list without padding.

    Actions: 0 up, 1 down, 2 left, 3 right.
    """

    actions: List[int] = []
    x, y = int(start_xy[0].item()), int(start_xy[1].item())
    tx, ty = int(target_xy[0].item()), int(target_xy[1].item())
    while x < tx:
        actions.append(3)
        x += 1
    while x > tx:
        actions.append(2)
        x -= 1
    while y < ty:
        actions.append(1)
        y += 1
    while y > ty:
        actions.append(0)
        y -= 1
    return actions


def _actions_tensor(actions: List[int], horizon: int, device: torch.device) -> torch.Tensor:
    if len(actions) >= horizon:
        return torch.tensor(actions[:horizon], dtype=torch.long, device=device)
    # There is no no-op action in this toy environment. Random padding is safer
    # than deterministic movement because success is measured as ever reached.
    pad = torch.randint(0, 4, (horizon - len(actions),), dtype=torch.long, device=device)
    if actions:
        return torch.cat([torch.tensor(actions, dtype=torch.long, device=device), pad], dim=0)
    return pad


def _path_actions(start_xy: torch.Tensor, target_xy: torch.Tensor, horizon: int, device: torch.device) -> torch.Tensor:
    return _actions_tensor(_path_action_list(start_xy, target_xy), horizon, device)


def _decode_initial_positions(initial_obs: torch.Tensor, size: int) -> torch.Tensor:
    scale = float(size - 1)
    return torch.round(initial_obs[:, :8].clamp(0.0, 1.0) * scale).long().clamp(0, size - 1)


def build_candidate_actions(initial_obs: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    """Build action candidates from random shooting plus visible-fact path priors.

    Candidate 0 is a deterministic key-then-goal path when it fits the horizon.
    Earlier versions accidentally padded the key path before appending the goal path,
    which made the planning benchmark much weaker than intended.
    """

    device = initial_obs.device
    batch_size = initial_obs.shape[0]
    candidates = torch.randint(0, cfg.num_actions, (batch_size, cfg.num_candidates, cfg.plan_horizon), device=device)
    pos = _decode_initial_positions(initial_obs, cfg.size)

    for b in range(batch_size):
        agent = pos[b, 0:2]
        key = pos[b, 2:4]
        door = pos[b, 4:6]
        goal = pos[b, 6:8]
        key_then_goal = _path_action_list(agent, key) + _path_action_list(key, goal)
        direct_goal = _path_action_list(agent, goal)
        key_then_door = _path_action_list(agent, key) + _path_action_list(key, door)
        door_then_goal = _path_action_list(agent, door) + _path_action_list(door, goal)

        if cfg.num_candidates >= 1:
            candidates[b, 0] = _actions_tensor(key_then_goal, cfg.plan_horizon, device)
        if cfg.num_candidates >= 2:
            candidates[b, 1] = _actions_tensor(direct_goal, cfg.plan_horizon, device)
        if cfg.num_candidates >= 3:
            candidates[b, 2] = _actions_tensor(key_then_door, cfg.plan_horizon, device)
        if cfg.num_candidates >= 4:
            candidates[b, 3] = _actions_tensor(door_then_goal, cfg.plan_horizon, device)
        if cfg.num_candidates >= 5:
            path = _path_action_list(agent, key)
            mixed = torch.randint(0, cfg.num_actions, (cfg.plan_horizon,), device=device)
            cut = min(cfg.plan_horizon, len(path))
            if cut > 0:
                mixed[:cut] = torch.tensor(path[:cut], dtype=torch.long, device=device)
            candidates[b, 4] = mixed
    return candidates


def _geometry_progress_score(flat_initial_obs: torch.Tensor, pred_next_obs: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    """Dense planning score from predicted position and visible t=0 facts.

    The learned model predicts future agent position and has_key. The planner is
    allowed to use the visible initial key/goal facts to prefer plans that move
    toward the key before pickup and toward the goal after pickup. This fixes the
    previous sparse-score failure mode where reward/done logits alone could not
    rank good candidates even when the candidate set contained them.
    """

    agent_xy = pred_next_obs[:, :2].clamp(0.0, 1.0)
    key_xy = flat_initial_obs[:, 2:4].clamp(0.0, 1.0)
    goal_xy = flat_initial_obs[:, 6:8].clamp(0.0, 1.0)
    key_prob = pred_next_obs[:, 8].clamp(0.0, 1.0)
    key_dist = (agent_xy - key_xy).abs().sum(dim=-1)
    goal_dist = (agent_xy - goal_xy).abs().sum(dim=-1)
    return -cfg.key_distance_weight * (1.0 - key_prob) * key_dist - cfg.goal_distance_weight * key_prob * goal_dist


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
        pred_next_obs = pred["next_obs"]
        done_prob = torch.sigmoid(pred["done_logit"])
        key_prob = pred_next_obs[:, 8].clamp(0.0, 1.0)
        score = score + pred["reward"] + _geometry_progress_score(flat_obs, pred_next_obs, cfg)
        max_done = torch.maximum(max_done, done_prob)
        max_key = torch.maximum(max_key, key_prob)
    score = score + pred["value"] + cfg.done_bonus * max_done + cfg.key_bonus * max_key
    return score.view(batch_size, num_candidates)


def score_candidates_gru(model: GRUWorldModel, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    batch_size, num_candidates, horizon = candidates.shape
    flat_initial_obs = initial_obs[:, None, :].expand(batch_size, num_candidates, cfg.max_obs_dim).reshape(batch_size * num_candidates, cfg.max_obs_dim)
    current_obs = flat_initial_obs
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
        score = score + pred["pred_reward"] + _geometry_progress_score(flat_initial_obs, current_obs, cfg)
        max_done = torch.maximum(max_done, done_prob)
        max_key = torch.maximum(max_key, key_prob)
    score = score + cfg.done_bonus * max_done + cfg.key_bonus * max_key
    return score.view(batch_size, num_candidates)


def score_candidates_mlp(model: MLPWorldModel, initial_obs: torch.Tensor, candidates: torch.Tensor, cfg: PlanningConfig) -> torch.Tensor:
    batch_size, num_candidates, horizon = candidates.shape
    flat_initial_obs = initial_obs[:, None, :].expand(batch_size, num_candidates, cfg.max_obs_dim).reshape(batch_size * num_candidates, cfg.max_obs_dim)
    current_obs = flat_initial_obs
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
        score = score + pred["pred_reward"] + _geometry_progress_score(flat_initial_obs, current_obs, cfg)
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


def cem_plan(
    model, label: str, initial_obs: torch.Tensor, cfg: PlanningConfig
) -> torch.Tensor:
    """CEM planning: iteratively refine candidates using world-model scoring.

    1. Generate random candidates
    2. Score → keep elite → perturb elites → repeat
    3. Return best candidate per batch item
    """
    B = initial_obs.shape[0]
    H = cfg.plan_horizon
    device = initial_obs.device

    candidates = torch.randint(0, cfg.num_actions, (B, cfg.num_candidates, H), device=device)

    n_elite = max(1, int(cfg.num_candidates * cfg.cem_elite_frac))
    n_new = cfg.num_candidates - n_elite

    for iteration in range(cfg.cem_iterations):
        scores = score_candidates(model, label, initial_obs, candidates, cfg)

        _, elite_idx = scores.topk(n_elite, dim=1)
        elite = torch.gather(candidates, 1, elite_idx.unsqueeze(-1).expand(-1, -1, H))

        noise_level = max(0.1, 1.0 - iteration / cfg.cem_iterations)
        noise_mask = torch.rand(B, n_new, H, device=device) < noise_level

        new_random = torch.randint(0, cfg.num_actions, (B, n_new, H), device=device)

        elite_tiled = elite[:, :n_new, :]
        if n_new > n_elite:
            repeats = (n_new + n_elite - 1) // n_elite
            elite_tiled = elite.repeat(1, repeats, 1)[:, :n_new, :]

        new_candidates = torch.where(noise_mask, new_random, elite_tiled)
        candidates = torch.cat([elite, new_candidates], dim=1)

    scores = score_candidates(model, label, initial_obs, candidates, cfg)
    best_idx = scores.argmax(dim=1)
    return candidates[torch.arange(B, device=device), best_idx]


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
    return {"total_reward": total_reward, "success": ever_done, "has_key": ever_key, "blocked_count": blocked_count}


def _expand_state_for_candidates(state: Dict[str, torch.Tensor], num_candidates: int) -> Dict[str, torch.Tensor]:
    return {key: value[:, None, ...].expand(value.shape[0], num_candidates, *value.shape[1:]).reshape(value.shape[0] * num_candidates, *value.shape[1:]) for key, value in state.items()}


def oracle_select_candidates(env: MemoryKeyDoorSequenceEnv, state: Dict[str, torch.Tensor], candidates: torch.Tensor, cfg: PlanningConfig) -> Dict[str, torch.Tensor]:
    batch_size, num_candidates, horizon = candidates.shape
    flat_state = _expand_state_for_candidates(state, num_candidates)
    flat_actions = candidates.reshape(batch_size * num_candidates, horizon)
    flat_result = execute_action_sequences(env, flat_state, flat_actions)
    true_score = flat_result["total_reward"] + cfg.done_bonus * flat_result["success"] + cfg.key_bonus * flat_result["has_key"]
    best_idx = true_score.view(batch_size, num_candidates).argmax(dim=1)
    best_actions = candidates[torch.arange(batch_size, device=candidates.device), best_idx]
    return execute_action_sequences(env, state, best_actions)


@torch.no_grad()
def evaluate_planning(models: Dict[str, object], cfg: PlanningConfig) -> List[Dict[str, float]]:
    torch.manual_seed(cfg.seed + 1000)
    device = torch.device(cfg.device)
    env = MemoryKeyDoorSequenceEnv(size=cfg.size, max_obs_dim=cfg.max_obs_dim)
    accum: Dict[str, Dict[str, List[float]]] = {}

    labels = ["random_policy", "oracle_candidate"] + list(models.keys())
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
        for key, value in (("success_rate", "success"), ("key_rate", "has_key"), ("avg_total_reward", "total_reward"), ("avg_blocked_count", "blocked_count")):
            accum["random_policy"][key].append(float(random_result[value].mean().item()))

        oracle_result = oracle_select_candidates(env, state, candidates, cfg)
        for key, value in (("success_rate", "success"), ("key_rate", "has_key"), ("avg_total_reward", "total_reward"), ("avg_blocked_count", "blocked_count")):
            accum["oracle_candidate"][key].append(float(oracle_result[value].mean().item()))

        for label, model in models.items():
            if cfg.use_cem:
                best_actions = cem_plan(model, label, initial_obs, cfg)
            else:
                scores = score_candidates(model, label, initial_obs, candidates, cfg)
                best_idx = scores.argmax(dim=1)
                best_actions = candidates[torch.arange(cfg.batch_size, device=device), best_idx]
            result = execute_action_sequences(env, state, best_actions)
            for key, value in (("success_rate", "success"), ("key_rate", "has_key"), ("avg_total_reward", "total_reward"), ("avg_blocked_count", "blocked_count")):
                accum[label][key].append(float(result[value].mean().item()))

    rows: List[Dict[str, float]] = []
    for label, metrics in accum.items():
        row: Dict[str, float] = {"model": label, "env": "memory_key_door_planning"}
        for key, values in metrics.items():
            row[key] = sum(values) / len(values)
        rows.append(row)
        print(f"[{label}] success={row['success_rate']:.2%} key={row['key_rate']:.2%} reward={row['avg_total_reward']:.4f} blocked={row['avg_blocked_count']:.4f}")
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
    parser.add_argument("--key-distance-weight", type=float, default=1.0)
    parser.add_argument("--goal-distance-weight", type=float, default=2.0)
    parser.add_argument("--allow-key-start", action="store_true")
    parser.add_argument("--use-cem", action="store_true")
    parser.add_argument("--cem-iterations", type=int, default=4)
    parser.add_argument("--cem-elite-frac", type=float, default=0.25)
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
        key_distance_weight=args.key_distance_weight,
        goal_distance_weight=args.goal_distance_weight,
        force_no_key_start=not args.allow_key_start,
        use_cem=args.use_cem,
        cem_iterations=args.cem_iterations,
        cem_elite_frac=args.cem_elite_frac,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    run(parse_args())
