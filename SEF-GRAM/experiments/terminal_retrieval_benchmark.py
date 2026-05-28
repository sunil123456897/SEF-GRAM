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


ALPHABET = "\n _ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:.[]-/"
PAD = 0
BOS = 1
EOS = 2
OFFSET = 3


@dataclass
class RetrievalConfig:
    latent_dim: int = 64
    train_steps: int = 600
    batch_size: int = 32
    distractor_len: int = 160
    eval_cases: int = 50
    lr: float = 3e-4
    context_lm_weight: float = 0.02
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


class RetrievalMemoryModel(nn.Module):
    def __init__(self, vocab_size: int, latent_dim: int):
        super().__init__()
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


def random_noise_line() -> str:
    templates = [
        "INFO time {a}:{b} worker alpha ok\n",
        "WARN retry {c} port {d}\n",
        "DEBUG cache hit {e}\n",
        "TRACE chunk {f} processed\n",
    ]
    return random.choice(templates).format(
        a=random.randint(10, 59),
        b=random.randint(10, 59),
        c=random.randint(0, 9),
        d=random.choice([3306, 5432, 6379, 8080]),
        e=random.choice(["yes", "no"]),
        f=random.randint(0, 99),
    )


def make_noise(length: int) -> str:
    text = ""
    while len(text) < length:
        text += random_noise_line()
    return text[:length]


def make_retrieval_task(distractor_len: int) -> Tuple[str, str, str]:
    value = str(random.randint(1000, 9999))
    prompt = (
        f"MEM_PORT {value}\n"
        "STATUS stored\n"
        f"{make_noise(distractor_len)}\n"
        "READ MEM_PORT\n"
        "ANSWER "
    )
    target = value + "\n"
    return prompt, target, prompt + target


def collate_strings(tokenizer: TinyCharTokenizer, samples: List[str], device: torch.device) -> torch.Tensor:
    encoded = [tokenizer.encode(sample, add_bos=True, add_eos=True) for sample in samples]
    max_len = max(len(x) for x in encoded)
    padded = [x + [PAD] * (max_len - len(x)) for x in encoded]
    return torch.tensor(padded, dtype=torch.long, device=device)


def collate_retrieval_batch(
    tokenizer: TinyCharTokenizer,
    tasks: List[Tuple[str, str, str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids = [tokenizer.encode(prompt, add_bos=True) for prompt, _, _ in tasks]
    target_ids = [tokenizer.encode(target) for _, target, _ in tasks]
    max_prompt_len = max(len(ids) for ids in prompt_ids)
    max_target_len = max(len(ids) for ids in target_ids)

    prompts = [ids + [PAD] * (max_prompt_len - len(ids)) for ids in prompt_ids]
    decoder_inputs = []
    decoder_targets = []
    for p_ids, t_ids in zip(prompt_ids, target_ids):
        inp = [p_ids[-1]] + t_ids[:-1]
        decoder_inputs.append(inp + [PAD] * (max_target_len - len(inp)))
        decoder_targets.append(t_ids + [PAD] * (max_target_len - len(t_ids)))

    return (
        torch.tensor(prompts, dtype=torch.long, device=device),
        torch.tensor(decoder_inputs, dtype=torch.long, device=device),
        torch.tensor(decoder_targets, dtype=torch.long, device=device),
    )


def retrieval_loss(
    model: RetrievalMemoryModel,
    tokenizer: TinyCharTokenizer,
    tasks: List[Tuple[str, str, str]],
    device: torch.device,
) -> torch.Tensor:
    prompts, decoder_inputs, decoder_targets = collate_retrieval_batch(tokenizer, tasks, device)
    _, memory = model(prompts)
    logits, _ = model(decoder_inputs, memory)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), decoder_targets.reshape(-1), ignore_index=PAD)


def context_lm_loss(
    model: RetrievalMemoryModel,
    tokenizer: TinyCharTokenizer,
    tasks: List[Tuple[str, str, str]],
    device: torch.device,
) -> torch.Tensor:
    full = [sample for _, _, sample in tasks]
    batch = collate_strings(tokenizer, full, device)
    logits, _ = model(batch[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), ignore_index=PAD)


def train_step(
    model: RetrievalMemoryModel,
    tokenizer: TinyCharTokenizer,
    tasks: List[Tuple[str, str, str]],
    device: torch.device,
    context_lm_weight: float = 0.02,
) -> torch.Tensor:
    loss = retrieval_loss(model, tokenizer, tasks, device)
    if context_lm_weight > 0:
        loss = loss + context_lm_weight * context_lm_loss(model, tokenizer, tasks, device)
    return loss


def generate_answer(
    model: RetrievalMemoryModel,
    tokenizer: TinyCharTokenizer,
    prompt: str,
    target_len: int,
    device: torch.device,
) -> str:
    ids = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    model.eval()
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


def evaluate(model: RetrievalMemoryModel, tokenizer: TinyCharTokenizer, cfg: RetrievalConfig, device: torch.device) -> Dict[str, float]:
    exact = 0
    matching = 0
    total = 0
    examples = []
    for i in range(cfg.eval_cases):
        prompt, target, _ = make_retrieval_task(cfg.distractor_len)
        pred = generate_answer(model, tokenizer, prompt, len(target), device)
        exact += int(pred.strip() == target.strip())
        for p_ch, t_ch in zip(pred, target):
            matching += int(p_ch == t_ch)
            total += 1
        if i < 3:
            examples.append((target.strip(), pred.strip()))
    return {
        "exact_retrieval_accuracy": exact / max(1, cfg.eval_cases),
        "char_accuracy": matching / max(1, total),
        "examples": examples,
    }


def run(cfg: RetrievalConfig) -> Dict[str, float]:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    tokenizer = TinyCharTokenizer()
    model = RetrievalMemoryModel(tokenizer.vocab_size, cfg.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    print(f"Terminal retrieval benchmark | latent_dim={cfg.latent_dim} | distractor={cfg.distractor_len} | device={device}")
    for step in range(1, cfg.train_steps + 1):
        tasks = [make_retrieval_task(random.randint(20, cfg.distractor_len)) for _ in range(cfg.batch_size)]
        model.train()
        optimizer.zero_grad()
        loss = train_step(model, tokenizer, tasks, device, context_lm_weight=cfg.context_lm_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.train_steps // 10) == 0:
            metrics = evaluate(model, tokenizer, cfg, device)
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"exact={metrics['exact_retrieval_accuracy']:.2%} "
                f"char={metrics['char_accuracy']:.2%} examples={metrics['examples']}"
            )

    final = evaluate(model, tokenizer, cfg, device)
    print("final:", final)
    return final


def parse_args() -> RetrievalConfig:
    parser = argparse.ArgumentParser(description="Focused SEF-GRAM terminal retrieval benchmark")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--distractor-len", type=int, default=160)
    parser.add_argument("--eval-cases", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context-lm-weight", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=11)
    return RetrievalConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    run(parse_args())
