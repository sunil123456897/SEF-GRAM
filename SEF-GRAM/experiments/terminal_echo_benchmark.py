from __future__ import annotations

from pathlib import Path
import argparse
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sef_gram.full_system import ExactEFLACell


ALPHABET = "\n $=>_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.[]-/"
PAD = 0
BOS = 1
EOS = 2
OFFSET = 3


@dataclass
class TerminalEchoConfig:
    latent_dim: int = 64
    train_steps: int = 600
    batch_size: int = 32
    distractor_len: int = 160
    eval_cases: int = 50
    lr: float = 3e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 11


class TinyCharTokenizer:
    def __init__(self):
        self.chars = ALPHABET
        self.stoi = {ch: i + OFFSET for i, ch in enumerate(self.chars)}
        self.itos = {i + OFFSET: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + OFFSET

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [self.stoi.get(ch, self.stoi["_"]) for ch in text]
        if add_bos:
            ids.insert(0, BOS)
        if add_eos:
            ids.append(EOS)
        return ids

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i >= OFFSET)


class EchoTerminalModel(nn.Module):
    """EFLA memory model for terminal-observation prediction."""

    def __init__(self, vocab_size: int, latent_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.cell = ExactEFLACell(latent_dim)
        self.head = nn.Linear(latent_dim, vocab_size)

    def initial_memory(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.latent_dim, self.latent_dim, device=device)

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = input_ids.shape
        if memory is None:
            memory = self.initial_memory(batch_size, input_ids.device)
        logits = []
        for t in range(steps):
            z = self.embedding(input_ids[:, t])
            z, memory = self.cell(z, memory)
            logits.append(self.head(z).unsqueeze(1))
        return torch.cat(logits, dim=1), memory


def random_log_line() -> str:
    templates = [
        "INFO [{time}] worker={worker} ok\n",
        "WARN [{time}] retry={retry} port={port}\n",
        "DEBUG [{time}] cache_hit={hit}\n",
        "ERROR [{time}] transient timeout port={port}\n",
    ]
    return random.choice(templates).format(
        time=f"12:{random.randint(10,59)}:{random.randint(10,59)}",
        worker=random.choice(["alpha", "beta", "gamma"]),
        retry=random.randint(0, 5),
        port=random.choice([3306, 5432, 6379, 8080]),
        hit=random.choice(["yes", "no"]),
    )


def make_distractor(length: int) -> str:
    text = ""
    while len(text) < length:
        text += random_log_line()
    return text[:length]


def make_terminal_task(distractor_len: int) -> Tuple[str, str, str]:
    value = str(random.randint(1000, 9999))
    var_name = "MEM_PORT"
    prompt = (
        f"$ set {var_name}={value}\n"
        f"> Variable {var_name} stored.\n"
        "$ cat system_logs.log\n"
        f"{make_distractor(distractor_len)}\n"
        f"$ print ${var_name}\n> "
    )
    target = value + "\n"
    return prompt, target, prompt + target


def collate(tokenizer: TinyCharTokenizer, samples: List[str], device: torch.device) -> torch.Tensor:
    encoded = [tokenizer.encode(sample, add_bos=True, add_eos=True) for sample in samples]
    max_len = max(len(x) for x in encoded)
    padded = [x + [PAD] * (max_len - len(x)) for x in encoded]
    return torch.tensor(padded, dtype=torch.long, device=device)


def train_step(model: EchoTerminalModel, tokenizer: TinyCharTokenizer, samples: List[str], device: torch.device) -> torch.Tensor:
    batch = collate(tokenizer, samples, device)
    logits, _ = model(batch[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), ignore_index=PAD)


def generate_answer(model: EchoTerminalModel, tokenizer: TinyCharTokenizer, prompt: str, target_len: int, device: torch.device) -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    with torch.no_grad():
        _, memory = model(ids)
        current = ids[:, -1:]
        generated: List[int] = []
        for _ in range(target_len):
            logits, memory = model(current, memory)
            next_id = int(logits[:, -1, :].argmax(dim=-1).item())
            generated.append(next_id)
            current = torch.tensor([[next_id]], dtype=torch.long, device=device)
    return tokenizer.decode(generated)


def evaluate(model: EchoTerminalModel, tokenizer: TinyCharTokenizer, cfg: TerminalEchoConfig, device: torch.device) -> Dict[str, float]:
    correct = 0
    for _ in range(cfg.eval_cases):
        prompt, target, _ = make_terminal_task(cfg.distractor_len)
        pred = generate_answer(model, tokenizer, prompt, len(target), device)
        if pred.strip() == target.strip():
            correct += 1
    return {"exact_retrieval_accuracy": correct / max(1, cfg.eval_cases)}


def run(cfg: TerminalEchoConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    tokenizer = TinyCharTokenizer()
    model = EchoTerminalModel(tokenizer.vocab_size, cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(f"Terminal ECHO benchmark | latent_dim={cfg.latent_dim} | distractor={cfg.distractor_len} | device={device}")
    for step in range(1, cfg.train_steps + 1):
        samples = [make_terminal_task(random.randint(20, cfg.distractor_len))[2] for _ in range(cfg.batch_size)]
        model.train()
        optimizer.zero_grad()
        loss = train_step(model, tokenizer, samples, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.train_steps // 10) == 0:
            metrics = evaluate(model, tokenizer, cfg, device)
            print(f"step={step:04d} loss={loss.item():.4f} exact={metrics['exact_retrieval_accuracy']:.2%}")

    final_metrics = evaluate(model, tokenizer, cfg, device)
    print("final:", final_metrics)
    return final_metrics


def parse_args() -> TerminalEchoConfig:
    parser = argparse.ArgumentParser(description="Full SEF-GRAM terminal ECHO memory benchmark")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distractor-len", type=int, default=160)
    parser.add_argument("--eval-cases", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=11)
    return TerminalEchoConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
