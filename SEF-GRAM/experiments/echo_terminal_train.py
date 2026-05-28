from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.echo_trainer import ECHOGDPOConfig, ECHOGRAMWrapper, train_echo_gdpo
from sef_gram.turn_based_env import MiniTerminalEnv


@torch.no_grad()
def evaluate_echo_model(model: ECHOGRAMWrapper, cfg: ECHOGDPOConfig, num_episodes: int = 100) -> Dict[str, float]:
    device = torch.device(cfg.device)
    env = MiniTerminalEnv(seed=cfg.seed + 9999)
    successes = 0
    turns_total = 0

    for _ in range(num_episodes):
        batch = env.reset(1, device)
        obs = batch.obs.reshape(-1, env.OBS_DIM)
        z, memory, _ = model.initial_state(obs)
        episode_reward = 0.0

        for turn in range(cfg.max_turns):
            logits = model.policy_logits(z)
            valid = batch.valid_actions.reshape(-1, env.NUM_ACTIONS)
            logits_masked = logits.clone()
            logits_masked[~valid] = -1e9
            action = logits_masked.argmax(dim=-1)

            batch = env.step(action.squeeze(-1) if action.shape[0] == 1 else action)
            episode_reward += float(batch.reward.reshape(-1)[0].item())
            turns_total += 1

            z_next, memory_next = model.forward_turn(z, memory, action)
            z, memory = z_next, memory_next

            if batch.done.reshape(-1)[0]:
                break

        if episode_reward > 0.5:
            successes += 1

    return {
        "success_rate": successes / num_episodes,
        "avg_turns": turns_total / num_episodes,
    }


def run(cfg: ECHOGDPOConfig, export_csv: str = "") -> List[Dict[str, float]]:
    model = train_echo_gdpo(cfg)
    eval_metrics = evaluate_echo_model(model, cfg)

    print("\n[echo_gdpo eval]")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value:.4f}")

    rows = [{"phase": "train", **eval_metrics}]
    if export_csv:
        path = Path(export_csv)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")

    return rows


def parse_args() -> Tuple[ECHOGDPOConfig, str]:
    parser = argparse.ArgumentParser(description="Turn-based ECHO-GDPO training on MiniTerminalEnv")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--echo-lambda", type=float, default=0.05)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--adamw-lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-csv", type=str, default="")
    args = parser.parse_args()

    cfg = ECHOGDPOConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        num_candidates=args.num_candidates,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        max_turns=args.max_turns,
        echo_lambda=args.echo_lambda,
        entropy_coef=args.entropy_coef,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        device=args.device,
        seed=args.seed,
    )
    return cfg, args.export_csv


if __name__ == "__main__":
    cfg, export_csv = parse_args()
    run(cfg, export_csv)
