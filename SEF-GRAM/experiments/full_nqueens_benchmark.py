from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.full_system import SEFGRAMObjective, SEFGRAMConfig, StochasticRecursiveWorldModel


Solution = Tuple[int, ...]


@dataclass
class NQueensRunConfig:
    board_size: int = 8
    latent_dim: int = 64
    hidden_dim: int = 128
    train_steps: int = 500
    batch_size: int = 64
    num_trajectories: int = 16
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7


def solve_nqueens(n: int) -> List[Solution]:
    """Return all N-Queens solutions as tuples of column indices per row."""

    solutions: List[Solution] = []
    cols = set()
    diag_a = set()
    diag_b = set()
    board: List[int] = []

    def backtrack(row: int) -> None:
        if row == n:
            solutions.append(tuple(board))
            return
        for col in range(n):
            if col in cols or (row - col) in diag_a or (row + col) in diag_b:
                continue
            cols.add(col)
            diag_a.add(row - col)
            diag_b.add(row + col)
            board.append(col)
            backtrack(row + 1)
            board.pop()
            diag_b.remove(row + col)
            diag_a.remove(row - col)
            cols.remove(col)

    backtrack(0)
    return solutions


def solution_is_valid(solution: Sequence[int], n: int) -> bool:
    if len(solution) != n:
        return False
    if any(col < 0 or col >= n for col in solution):
        return False
    if len(set(solution)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(solution[i] - solution[j]) == abs(i - j):
                return False
    return True


def make_context(solution: Solution, n: int) -> torch.Tensor:
    """Compact context: board size plus normalized solution statistics.

    This is deliberately not the solution itself. The target solution is used only as
    teacher-forced action sequence during training.
    """

    x = torch.zeros(n, dtype=torch.float32)
    x[0] = n / 16.0
    x[1] = sum(solution) / max(1.0, n * (n - 1))
    x[2] = sum((i + 1) * (c + 1) for i, c in enumerate(solution)) / max(1.0, n * n * n)
    x[3] = len(set(solution)) / n
    for i in range(4, n):
        x[i] = ((i + 1) * 0.137) % 1.0
    return x


def batch_from_solutions(solutions: List[Solution], n: int, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    batch = [random.choice(solutions) for _ in range(batch_size)]
    context = torch.stack([make_context(sol, n) for sol in batch], dim=0).to(device)
    actions = torch.tensor(batch, dtype=torch.long, device=device)
    return context, actions


def teacher_forced_policy_loss(model: StochasticRecursiveWorldModel, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    rollout = model.recursive_rollout(
        context,
        depth=actions.shape[1],
        num_trajectories=1,
        action_ids=actions,
        sample_actions=False,
    )
    logits = rollout["policy_logits"].squeeze(1)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), actions.reshape(-1))


def evaluate(model: StochasticRecursiveWorldModel, n: int, num_trajectories: int, device: torch.device) -> Dict[str, float]:
    context = torch.zeros(1, n, dtype=torch.float32, device=device)
    context[0, 0] = n / 16.0
    model.eval()
    with torch.no_grad():
        rollout = model.recursive_rollout(
            context,
            depth=n,
            num_trajectories=num_trajectories,
            sample_actions=True,
        )
    actions = rollout["actions"][0].cpu().tolist()
    candidates = [tuple(map(int, seq)) for seq in actions]
    unique_candidates = set(candidates)
    valid = [candidate for candidate in candidates if solution_is_valid(candidate, n)]
    unique_valid = set(valid)

    return {
        "valid_solution_rate": len(valid) / max(1, len(candidates)),
        "unique_solution_count": float(len(unique_valid)),
        "trajectory_diversity": len(unique_candidates) / max(1, len(candidates)),
        "best_of_k_success": 1.0 if valid else 0.0,
    }


def run(cfg: NQueensRunConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    solutions = solve_nqueens(cfg.board_size)
    if not solutions:
        raise RuntimeError(f"No solutions found for N={cfg.board_size}")

    model = StochasticRecursiveWorldModel(
        SEFGRAMConfig(
            input_dim=cfg.board_size,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.board_size,
            recursion_depth=cfg.board_size,
            num_trajectories=cfg.num_trajectories,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(f"N-Queens full benchmark | N={cfg.board_size} | known_solutions={len(solutions)} | device={device}")
    for step in range(1, cfg.train_steps + 1):
        model.train()
        context, actions = batch_from_solutions(solutions, cfg.board_size, cfg.batch_size, device)
        optimizer.zero_grad()
        loss = teacher_forced_policy_loss(model, context, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.train_steps // 10) == 0:
            metrics = evaluate(model, cfg.board_size, cfg.num_trajectories, device)
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"valid={metrics['valid_solution_rate']:.2%} "
                f"unique_valid={metrics['unique_solution_count']:.0f} "
                f"diversity={metrics['trajectory_diversity']:.2%} "
                f"best_of_k={metrics['best_of_k_success']:.0f}"
            )

    final_metrics = evaluate(model, cfg.board_size, cfg.num_trajectories, device)
    print("final:", final_metrics)
    return final_metrics


def parse_args() -> NQueensRunConfig:
    parser = argparse.ArgumentParser(description="Full SEF-GRAM N-Queens multi-trajectory benchmark")
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-trajectories", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    return NQueensRunConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
