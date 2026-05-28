from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.echo_trainer import (
    ECHOGDPOConfig,
    ECHOGRAMWrapper,
    _gdpo_advantages_from_rewards,
)
from sef_gram.full_system import diagonal_gaussian_nll
from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.turn_based_env import TwoPhaseMemoryEnv
from sef_gram.turn_baselines import MLPTurnBaseline, GRUTurnBaseline, LSTMTurnBaseline
from sef_gram.utils import compute_dirichlet_energy


def _gdpo_step_for_model(
    model: nn.Module,
    env,
    cfg: ECHOGDPOConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    B = cfg.batch_size
    K = cfg.num_candidates

    batch = env.reset(B, device, num_candidates=K)
    obs = batch.obs
    z, memory, _ = model.initial_state(obs)

    all_logits = []
    all_actions = []
    all_env_mu = []
    all_env_logvar = []
    all_target_mu = []
    all_target_logvar = []
    all_env_logits = []
    all_env_targets = []
    all_rewards = []
    use_latent_echo = hasattr(model, 'predict_env_latent')

    for _ in range(cfg.max_turns):
        logits = model.policy_logits(z)
        dist = torch.distributions.Categorical(logits=logits)
        actions_flat = dist.sample()

        actions_bk = actions_flat.view(B, K)
        env_batch = env.step(actions_bk)

        z_r = z.view(B, K, -1)
        z_avg = z_r.mean(dim=1)

        if use_latent_echo:
            pred_mu, pred_logvar = model.predict_env_latent(z_avg)
            next_obs_avg = env_batch.obs.reshape(B, K, -1).mean(dim=1)
            target_mu, target_logvar = model.encode_env_target(next_obs_avg)
            all_env_mu.append(pred_mu)
            all_env_logvar.append(pred_logvar)
            all_target_mu.append(target_mu)
            all_target_logvar.append(target_logvar)
        else:
            env_logits = model.env_logits_from_z(z_avg)
            if env_logits is None:
                env_logits = torch.zeros(B, getattr(env, 'NUM_ENV_CLASSES', 6), device=device)
            all_env_logits.append(env_logits)
            all_env_targets.append(env_batch.env_target)

        next_obs_flat = env_batch.obs.reshape(-1, env_batch.obs.shape[-1])
        z_next, memory_next = model.forward_turn(z, memory, actions_flat, obs=next_obs_flat)

        all_logits.append(logits)
        all_actions.append(actions_flat)
        all_rewards.append(env_batch.reward)

        z = z_next
        memory = memory_next

        if env_batch.done.any():
            next_obs = env_batch.obs.reshape(-1, env_batch.obs.shape[-1])
            h_new, mem_new, _ = model.initial_state(next_obs)
            curr_done = env_batch.done.reshape(-1)
            z = torch.where(curr_done.unsqueeze(-1), h_new, z)
            mem_shape = [1] * (memory.dim() - 1)
            memory = torch.where(curr_done.view(-1, *mem_shape), mem_new, memory)

    actions_t = torch.stack(all_actions, dim=1)
    logits_t = torch.stack(all_logits, dim=1)
    rewards_t = torch.stack(all_rewards, dim=-1)

    traj_rewards = rewards_t.sum(dim=-1)
    advantages_expanded = _gdpo_advantages_from_rewards(traj_rewards)
    advantages = advantages_expanded.reshape(-1)

    log_probs = nn.functional.log_softmax(logits_t, dim=-1).gather(-1, actions_t.long().unsqueeze(-1)).squeeze(-1)
    policy_loss = -(log_probs.sum(dim=-1) * advantages.to(logits_t.device)).mean()

    probs = nn.functional.softmax(logits_t, dim=-1)
    entropy = -(probs * nn.functional.log_softmax(logits_t, dim=-1)).sum(dim=-1).mean()

    if use_latent_echo:
        env_mu_t = torch.stack(all_env_mu, dim=1)
        env_logvar_t = torch.stack(all_env_logvar, dim=1)
        target_mu_t = torch.stack(all_target_mu, dim=1).detach()
        target_logvar_t = torch.stack(all_target_logvar, dim=1).detach()
        env_latent_dim = env_mu_t.shape[-1]
        env_loss = diagonal_gaussian_nll(
            target_mu_t.reshape(-1, env_latent_dim),
            env_mu_t.reshape(-1, env_latent_dim),
            env_logvar_t.reshape(-1, env_latent_dim),
        )
    else:
        env_logits_t = torch.stack(all_env_logits, dim=1)
        env_targets_t = torch.stack(all_env_targets, dim=-1)
        env_target_mask = env_targets_t > 0
        if env_target_mask.any():
            env_logits_expanded = env_logits_t.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, env_logits_t.shape[-1])
            env_loss = nn.functional.cross_entropy(
                env_logits_expanded.reshape(-1, env_logits_expanded.shape[-1]),
                env_targets_t.reshape(-1).long(),
            )
        else:
            env_loss = torch.tensor(0.0, device=device)

    total = policy_loss + cfg.echo_lambda * env_loss - cfg.entropy_coef * entropy

    de = compute_dirichlet_energy(z.view(-1, z.shape[-1]))
    success_rate = (traj_rewards.max(dim=-1).values > 0.5).float().mean().item()

    return total, {
        "total": float(total.item()),
        "policy_loss": float(policy_loss.item()),
        "env_loss": float(env_loss.item()),
        "entropy": float(entropy.item()),
        "success_rate": success_rate,
        "mean_reward": float(traj_rewards.mean().item()),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module, env, cfg: ECHOGDPOConfig, device: torch.device, num_episodes: int = 200,
) -> Dict[str, float]:
    successes = 0
    turns_total = 0

    for _ in range(num_episodes):
        batch = env.reset(1, device)
        obs = batch.obs
        z, memory, _ = model.initial_state(obs)
        ep_reward = 0.0

        for turn in range(cfg.max_turns):
            logits = model.policy_logits(z)
            valid = batch.valid_actions.reshape(-1, env.NUM_ACTIONS)
            logits_masked = logits.clone()
            logits_masked[~valid] = -1e9
            action = logits_masked.argmax(dim=-1)

            batch = env.step(action)
            turns_total += 1

            z_next, memory_next = model.forward_turn(z, memory, action, obs=batch.obs.reshape(-1, batch.obs.shape[-1]))
            z, memory = z_next, memory_next

            if batch.done.reshape(-1)[0]:
                if batch.reward.reshape(-1)[0].item() > 0.5:
                    successes += 1
                break

    return {"success_rate": successes / num_episodes, "avg_turns": turns_total / num_episodes, "env_accuracy": 0.0}


def train_one_model(
    model: nn.Module,
    env_factory,
    cfg: ECHOGDPOConfig,
    device: torch.device,
    label: str,
    track_steps: bool = True,
) -> Tuple[nn.Module, List[Dict[str, float]]]:
    from sef_gram.optimization import MuonWithAuxAdam

    is_echo = isinstance(model, ECHOGRAMWrapper)
    if is_echo:
        optimizer = MuonWithAuxAdam(
            model.parameters(), lr=cfg.muon_lr, momentum=cfg.muon_momentum,
            ns_steps=cfg.muon_ns_steps, adamw_lr=cfg.adamw_lr,
            adamw_betas=cfg.adamw_betas, adamw_wd=cfg.adamw_wd,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.adamw_lr, weight_decay=cfg.adamw_wd)

    tracking: List[Dict[str, float]] = []
    log_every = max(1, cfg.steps // 20)

    t0 = time.perf_counter()
    for step in range(1, cfg.steps + 1):
        env = env_factory()
        model.train()
        optimizer.zero_grad()
        loss, metrics = _gdpo_step_for_model(model, env, cfg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if isinstance(model, ECHOGRAMWrapper):
            model.update_target_encoder()

        if track_steps and (step % log_every == 0 or step == 1):
            row = {f"{label}_{k}": v for k, v in metrics.items()}
            row["step"] = step
            row["model"] = label
            tracking.append(row)

    elapsed = time.perf_counter() - t0
    final_row = tracking[-1] if tracking else {}
    final_row["wall_time_s"] = elapsed
    final_row["steps_per_sec"] = cfg.steps / elapsed if elapsed > 0 else 0.0
    return model, tracking


@dataclass
class CompareConfig:
    seeds: Tuple[int, ...] = (42, 43, 44)
    steps: int = 500
    batch_size: int = 32
    num_candidates: int = 4
    max_turns: int = 6
    echo_lambda: float = 0.05
    entropy_coef: float = 0.01
    latent_dim: int = 64
    hidden_dim: int = 128
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adamw_lr: float = 3e-4
    adamw_betas: Tuple[float, float] = (0.9, 0.95)
    adamw_wd: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    export_csv: str = ""
    export_tracking_csv: str = ""


def build_model(label: str, env, cfg: CompareConfig, device: torch.device) -> nn.Module:
    if label == "echo_gram":
        return ECHOGRAMWrapper(
            obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS,
            env_vocab_size=env.NUM_ENV_CLASSES,
            latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
        ).to(device)
    elif label == "lstm":
        return LSTMTurnBaseline(
            obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS,
            env_vocab_size=env.NUM_ENV_CLASSES, hidden_dim=cfg.hidden_dim,
        ).to(device)
    elif label == "gru":
        return GRUTurnBaseline(
            obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS,
            env_vocab_size=env.NUM_ENV_CLASSES, hidden_dim=cfg.hidden_dim,
        ).to(device)
    else:
        return MLPTurnBaseline(
            obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS,
            env_vocab_size=env.NUM_ENV_CLASSES, hidden_dim=cfg.hidden_dim,
        ).to(device)


def run_compare(cfg: CompareConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    device = torch.device(cfg.device)
    models_order = ["echo_gram", "lstm", "gru", "mlp"]
    summary_rows: List[Dict[str, float]] = []
    all_tracking: List[Dict[str, float]] = []

    print(f"[compare] env=TwoPhaseMemory seeds={cfg.seeds} steps={cfg.steps} batch={cfg.batch_size} candidates={cfg.num_candidates} device={device}")
    print()

    train_cfg = ECHOGDPOConfig(
        steps=cfg.steps, batch_size=cfg.batch_size, num_candidates=cfg.num_candidates,
        latent_dim=cfg.latent_dim, hidden_dim=cfg.hidden_dim,
        max_turns=cfg.max_turns, echo_lambda=cfg.echo_lambda,
        entropy_coef=cfg.entropy_coef, muon_lr=cfg.muon_lr,
        adamw_lr=cfg.adamw_lr, device=cfg.device,
    )

    for label in models_order:
        seed_success = []
        seed_turns = []
        seed_env_acc = []
        seed_time = []

        for seed in cfg.seeds:
            torch.manual_seed(seed)
            env_factory = lambda s=seed: TwoPhaseMemoryEnv(seed=s + 10000)
            eval_env = TwoPhaseMemoryEnv(seed=seed + 20000)
            model = build_model(label, eval_env, cfg, device)

            print(f"[{label}] seed={seed}")
            model, tracking = train_one_model(model, env_factory, train_cfg, device, label)

            eval_metrics = evaluate_model(model, eval_env, train_cfg, device)
            seed_success.append(eval_metrics["success_rate"])
            seed_turns.append(eval_metrics["avg_turns"])
            seed_env_acc.append(eval_metrics.get("env_accuracy", 0.0))
            seed_time.append(tracking[-1].get("wall_time_s", 0.0) if tracking else 0.0)

            for row in tracking:
                row["seed"] = seed
            all_tracking.extend(tracking)

            print(f"  success={eval_metrics['success_rate']:.3f} turns={eval_metrics['avg_turns']:.2f} env_acc={eval_metrics.get('env_accuracy', 0):.3f}")

        import statistics
        print(f"[{label}] MEAN success={statistics.mean(seed_success):.3f}±{statistics.stdev(seed_success):.3f} turns={statistics.mean(seed_turns):.2f} time={statistics.mean(seed_time):.1f}s")

        summary_rows.append({
            "model": label,
            "success_mean": statistics.mean(seed_success),
            "success_std": statistics.stdev(seed_success) if len(seed_success) > 1 else 0.0,
            "turns_mean": statistics.mean(seed_turns),
            "env_acc_mean": statistics.mean(seed_env_acc),
            "time_mean_s": statistics.mean(seed_time),
            "num_seeds": len(cfg.seeds),
        })
        print()

    print("[compare summary]")
    print(f"{'model':<12} {'success':>10} {'turns':>8} {'env_acc':>9} {'time(s)':>9}")
    print("-" * 52)
    for row in summary_rows:
        print(
            f"{row['model']:<12} {row['success_mean']:>8.3f}±{row['success_std']:.3f} "
            f"{row['turns_mean']:>7.2f} {row['env_acc_mean']:>8.3f} {row['time_mean_s']:>8.1f}"
        )

    if cfg.export_tracking_csv and all_tracking:
        all_keys = set()
        for row in all_tracking:
            all_keys.update(row.keys())
        path = Path(cfg.export_tracking_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(all_tracking)
        print(f"\nwrote tracking: {path}")

    if cfg.export_csv and summary_rows:
        path = Path(cfg.export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"wrote summary: {path}")

    return summary_rows, all_tracking


def parse_args() -> CompareConfig:
    parser = argparse.ArgumentParser(description="ECHO-GRAM vs MLP vs GRU comparison on MiniTerminalEnv")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--echo-lambda", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--export-csv", type=str, default="")
    parser.add_argument("--export-tracking-csv", type=str, default="")
    args = parser.parse_args()

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    return CompareConfig(
        seeds=seeds, steps=args.steps, batch_size=args.batch_size,
        num_candidates=args.num_candidates, max_turns=args.max_turns,
        echo_lambda=args.echo_lambda, entropy_coef=args.entropy_coef,
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        muon_lr=args.muon_lr, adamw_lr=args.adamw_lr,
        device=args.device, export_csv=args.export_csv,
        export_tracking_csv=args.export_tracking_csv,
    )


if __name__ == "__main__":
    run_compare(parse_args())
