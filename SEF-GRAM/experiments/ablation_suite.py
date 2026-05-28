from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.full_nqueens_benchmark import (
    NQueensRunConfig,
    batch_from_solutions,
    evaluate as evaluate_nqueens_once,
    solve_nqueens,
    teacher_forced_policy_loss,
)
from experiments.terminal_retrieval_benchmark import (
    RetrievalConfig,
    RetrievalMemoryModel,
    TinyCharTokenizer,
    collate_retrieval_batch,
    evaluate as evaluate_hybrid_retrieval,
    make_retrieval_task,
)
from sef_gram.full_system import ExactEFLACell, SEFGRAMConfig, StochasticRecursiveWorldModel


@dataclass
class AblationConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 7
    board_size: int = 8
    nqueens_steps: int = 300
    nqueens_batch_size: int = 64
    nqueens_eval_trials: int = 20
    nqueens_k_values: Tuple[int, ...] = (1, 4, 16)
    terminal_steps: int = 200
    terminal_batch_size: int = 32
    terminal_distractor_len: int = 80
    terminal_eval_cases: int = 50
    latent_dim: int = 64
    hidden_dim: int = 128
    lr: float = 3e-4


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [key for key in rows[0].keys() if isinstance(rows[0][key], (int, float))]
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def train_nqueens_model(cfg: AblationConfig) -> StochasticRecursiveWorldModel:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    solutions = solve_nqueens(cfg.board_size)
    model = StochasticRecursiveWorldModel(
        SEFGRAMConfig(
            input_dim=cfg.board_size,
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.hidden_dim,
            num_actions=cfg.board_size,
            recursion_depth=cfg.board_size,
            num_trajectories=max(cfg.nqueens_k_values),
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(
        f"[nqueens] training N={cfg.board_size} steps={cfg.nqueens_steps} "
        f"solutions={len(solutions)} device={device}"
    )
    for step in range(1, cfg.nqueens_steps + 1):
        context, actions = batch_from_solutions(solutions, cfg.board_size, cfg.nqueens_batch_size, device)
        model.train()
        optimizer.zero_grad()
        loss = teacher_forced_policy_loss(model, context, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, cfg.nqueens_steps // 5) == 0:
            print(f"[nqueens] step={step:04d} loss={loss.item():.4f}")
    return model


def run_nqueens_k_ablation(cfg: AblationConfig) -> Dict[int, Dict[str, float]]:
    device = torch.device(cfg.device)
    model = train_nqueens_model(cfg)
    results: Dict[int, Dict[str, float]] = {}
    print("\n[nqueens] K ablation")
    for k in cfg.nqueens_k_values:
        rows = [evaluate_nqueens_once(model, cfg.board_size, k, device) for _ in range(cfg.nqueens_eval_trials)]
        metrics = mean_metrics(rows)
        results[int(k)] = metrics
        print(
            f"K={k:<3} valid={metrics['valid_solution_rate']:.2%} "
            f"unique_valid={metrics['unique_solution_count']:.2f} "
            f"diversity={metrics['trajectory_diversity']:.2%} "
            f"best_of_k={metrics['best_of_k_success']:.2%}"
        )
    return results


class NeuralOnlyRetrievalModel(nn.Module):
    """Neural-only terminal retrieval baseline without explicit KV parser."""

    def __init__(self, vocab_size: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, latent_dim, padding_idx=0)
        self.cell = ExactEFLACell(latent_dim)
        self.readout = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, 4 * 10),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, steps = input_ids.shape
        memory = torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=input_ids.device)
        z = torch.zeros(batch_size, self.latent_dim, device=input_ids.device)
        for t in range(steps):
            token_ids = input_ids[:, t]
            active = (token_ids != 0).view(batch_size, 1)
            z_next, memory_next = self.cell(self.embedding(token_ids), memory)
            z = torch.where(active, z_next, z)
            memory = torch.where(active.view(batch_size, 1, 1), memory_next, memory)
        features = torch.cat([z, memory.mean(dim=1), torch.diagonal(memory, dim1=1, dim2=2)], dim=-1)
        return self.readout(features).view(batch_size, 4, 10)


def neural_retrieval_train_step(
    model: NeuralOnlyRetrievalModel,
    tokenizer: TinyCharTokenizer,
    tasks: List[Tuple[str, str, str]],
    device: torch.device,
) -> torch.Tensor:
    prompts, digit_targets = collate_retrieval_batch(tokenizer, tasks, device)
    logits = model(prompts)
    return F.cross_entropy(logits.reshape(-1, 10), digit_targets.reshape(-1))


def evaluate_neural_retrieval(
    model: NeuralOnlyRetrievalModel,
    tokenizer: TinyCharTokenizer,
    cfg: AblationConfig,
    device: torch.device,
) -> Dict[str, float]:
    exact = 0
    matching = 0
    total = 0
    examples = []
    model.eval()
    for i in range(cfg.terminal_eval_cases):
        prompt, target, _ = make_retrieval_task(cfg.terminal_distractor_len)
        prompt_ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
        with torch.no_grad():
            pred_digits = model(prompt_ids).argmax(dim=-1)[0].tolist()
        pred = "".join(str(int(d)) for d in pred_digits)
        target_clean = target.strip()
        exact += int(pred == target_clean)
        for p_ch, t_ch in zip(pred, target_clean):
            matching += int(p_ch == t_ch)
            total += 1
        if i < 3:
            examples.append((target_clean, pred))
    return {
        "exact_retrieval_accuracy": exact / max(1, cfg.terminal_eval_cases),
        "char_accuracy": matching / max(1, total),
        "examples": examples,
    }


def run_terminal_retrieval_ablation(cfg: AblationConfig) -> Dict[str, Dict[str, float]]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    tokenizer = TinyCharTokenizer()

    neural = NeuralOnlyRetrievalModel(tokenizer.vocab_size, cfg.latent_dim).to(device)
    neural_optimizer = torch.optim.AdamW(neural.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(
        f"\n[terminal] neural_only training steps={cfg.terminal_steps} "
        f"distractor={cfg.terminal_distractor_len} device={device}"
    )
    for step in range(1, cfg.terminal_steps + 1):
        tasks = [make_retrieval_task(random.randint(20, cfg.terminal_distractor_len)) for _ in range(cfg.terminal_batch_size)]
        neural.train()
        neural_optimizer.zero_grad()
        loss = neural_retrieval_train_step(neural, tokenizer, tasks, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(neural.parameters(), 1.0)
        neural_optimizer.step()
        if step == 1 or step % max(1, cfg.terminal_steps // 5) == 0:
            metrics = evaluate_neural_retrieval(neural, tokenizer, cfg, device)
            print(
                f"[terminal:neural] step={step:04d} loss={loss.item():.4f} "
                f"exact={metrics['exact_retrieval_accuracy']:.2%} char={metrics['char_accuracy']:.2%} "
                f"examples={metrics['examples']}"
            )

    neural_metrics = evaluate_neural_retrieval(neural, tokenizer, cfg, device)

    hybrid_cfg = RetrievalConfig(
        latent_dim=cfg.latent_dim,
        train_steps=1,
        batch_size=cfg.terminal_batch_size,
        distractor_len=cfg.terminal_distractor_len,
        eval_cases=cfg.terminal_eval_cases,
        lr=cfg.lr,
        context_lm_weight=0.0,
        device=cfg.device,
        seed=cfg.seed,
    )
    hybrid = RetrievalMemoryModel(tokenizer.vocab_size, cfg.latent_dim).to(device)
    hybrid_metrics = evaluate_hybrid_retrieval(hybrid, tokenizer, hybrid_cfg, device)

    print("\n[terminal] ablation summary")
    print(
        f"neural_only       exact={neural_metrics['exact_retrieval_accuracy']:.2%} "
        f"char={neural_metrics['char_accuracy']:.2%} examples={neural_metrics['examples']}"
    )
    print(
        f"hybrid_kv_parser  exact={hybrid_metrics['exact_retrieval_accuracy']:.2%} "
        f"char={hybrid_metrics['char_accuracy']:.2%} examples={hybrid_metrics['examples']}"
    )
    return {"neural_only": neural_metrics, "hybrid_kv_parser": hybrid_metrics}


def run(cfg: AblationConfig) -> Dict[str, object]:
    nqueens = run_nqueens_k_ablation(cfg)
    terminal = run_terminal_retrieval_ablation(cfg)
    return {"nqueens_k_ablation": nqueens, "terminal_retrieval_ablation": terminal}


def parse_args() -> AblationConfig:
    parser = argparse.ArgumentParser(description="SEF-GRAM ablation suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--nqueens-steps", type=int, default=300)
    parser.add_argument("--nqueens-batch-size", type=int, default=64)
    parser.add_argument("--nqueens-eval-trials", type=int, default=20)
    parser.add_argument("--nqueens-k-values", type=str, default="1,4,16")
    parser.add_argument("--terminal-steps", type=int, default=200)
    parser.add_argument("--terminal-batch-size", type=int, default=32)
    parser.add_argument("--terminal-distractor-len", type=int, default=80)
    parser.add_argument("--terminal-eval-cases", type=int, default=50)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    return AblationConfig(
        device=args.device,
        seed=args.seed,
        board_size=args.board_size,
        nqueens_steps=args.nqueens_steps,
        nqueens_batch_size=args.nqueens_batch_size,
        nqueens_eval_trials=args.nqueens_eval_trials,
        nqueens_k_values=tuple(int(part.strip()) for part in args.nqueens_k_values.split(",") if part.strip()),
        terminal_steps=args.terminal_steps,
        terminal_batch_size=args.terminal_batch_size,
        terminal_distractor_len=args.terminal_distractor_len,
        terminal_eval_cases=args.terminal_eval_cases,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )


if __name__ == "__main__":
    run(parse_args())
