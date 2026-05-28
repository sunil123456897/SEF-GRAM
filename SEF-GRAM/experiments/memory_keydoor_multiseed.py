from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import sys
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.memory_keydoor_compare import MemoryKeyDoorConfig, run as run_memory_keydoor


@dataclass
class MultiSeedMemoryConfig:
    seeds: Tuple[int, ...] = (37, 38, 39)
    base: MemoryKeyDoorConfig = MemoryKeyDoorConfig()
    export_csv: str = ""
    export_raw_csv: str = ""


def parse_seed_list(text: str) -> Tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _numeric_keys(rows: Iterable[Dict[str, object]]) -> List[str]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if key in {"seed", "horizon"}:
                continue
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def _mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [float(v) for v in values if not math.isnan(float(v))]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return mean, math.sqrt(var)


def aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["model"]), str(row["env"])), []).append(row)

    out: List[Dict[str, object]] = []
    numeric_keys = _numeric_keys(rows)
    for (model, env), group in sorted(groups.items()):
        agg: Dict[str, object] = {"model": model, "env": env, "n_seeds": len(group)}
        for key in numeric_keys:
            values = [float(row[key]) for row in group if key in row and isinstance(row[key], (int, float))]
            mean, std = _mean_std(values)
            agg[f"{key}_mean"] = mean
            agg[f"{key}_std"] = std
        out.append(agg)
    return out


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    all_keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def run(cfg: MultiSeedMemoryConfig) -> List[Dict[str, object]]:
    raw_rows: List[Dict[str, object]] = []
    for seed in cfg.seeds:
        print(f"\n[multi-seed] running seed={seed}")
        seed_cfg = replace(cfg.base, seed=seed, export_csv="")
        rows = run_memory_keydoor(seed_cfg)
        for row in rows:
            row = dict(row)
            row["seed"] = seed
            raw_rows.append(row)

    aggregate = aggregate_rows(raw_rows)
    print("\n[multi-seed aggregate summary]")
    for row in aggregate:
        print(
            f"{row['model']}:{row['env']} "
            f"obs={row.get('rollout_obs_mse_avg_mean', float('nan')):.4f}±{row.get('rollout_obs_mse_avg_std', float('nan')):.4f} "
            f"reward={row.get('rollout_reward_mse_avg_mean', float('nan')):.4f}±{row.get('rollout_reward_mse_avg_std', float('nan')):.4f} "
            f"done_bce={row.get('rollout_done_bce_avg_mean', float('nan')):.4f}±{row.get('rollout_done_bce_avg_std', float('nan')):.4f} "
            f"has_key={row.get('has_key_accuracy_mean', float('nan')):.2%}±{row.get('has_key_accuracy_std', float('nan')):.2%} "
            f"blocked={row.get('blocked_accuracy_mean', float('nan')):.2%}±{row.get('blocked_accuracy_std', float('nan')):.2%}"
        )

    if cfg.export_raw_csv:
        write_csv(cfg.export_raw_csv, raw_rows)
    if cfg.export_csv:
        write_csv(cfg.export_csv, aggregate)
    return aggregate


def parse_args() -> MultiSeedMemoryConfig:
    parser = argparse.ArgumentParser(description="Multi-seed partially observable key-door memory benchmark")
    parser.add_argument("--seeds", type=str, default="37,38,39")
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
    parser.add_argument("--device", type=str, default=MemoryKeyDoorConfig.device)
    parser.add_argument("--export-csv", type=str, default="")
    parser.add_argument("--export-raw-csv", type=str, default="")
    args = parser.parse_args()
    base = MemoryKeyDoorConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        rollout_horizon=args.rollout_horizon,
        size=args.size,
        max_obs_dim=args.max_obs_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        lr=args.lr,
        device=args.device,
        seed=0,
        export_csv="",
    )
    return MultiSeedMemoryConfig(
        seeds=parse_seed_list(args.seeds),
        base=base,
        export_csv=args.export_csv,
        export_raw_csv=args.export_raw_csv,
    )


if __name__ == "__main__":
    run(parse_args())
