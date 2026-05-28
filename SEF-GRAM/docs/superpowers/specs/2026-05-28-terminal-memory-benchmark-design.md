# Design Specification: Character Tokenizer and Terminal Memory Capacity Stress-Test

## Goal
Implement a robust character-level tokenization system (ASCII-260 vocabulary) and a comprehensive stress-testing benchmark ("Test 1: Terminal Memory Capacity Benchmark") to evaluate the capacity, long-context retention, and associative retrieval limits of the continuous $S_t \in \mathbb{R}^{D \times D}$ ODE memory matrix in `EFLACell`.

The benchmark evaluates two scenarios:
1. **Synthetic Controlled Memory (Approach 1)**: Highly controlled synthetic terminal histories storing key-value pairs, introducing variable-length distractors ($L$), and querying variables to determine the exact retention capacity of $S_t$ vs. context length and memory load.
2. **Real Terminal Logs (Approach 2)**: Real terminal command-response logs streamed from the `obiwan96/endless-terminals` Hugging Face dataset (with a robust offline procedural fallback of git, docker, python, and bash sessions) to assess character-level cross-entropy (perplexity).

---

## Architecture Design

### 1. Character Tokenizer (`sef_gram/tokenizer.py`)
A highly optimized, robust character-level tokenizer mapped to a fixed dictionary of size 260:
- Special Tokens:
  - `0`: `<pad>`
  - `1`: `<bos>`
  - `2`: `<eos>`
  - `3`: `<unk>`
- ASCII characters (0-255): Mapped to IDs `4` to `259` using `ord(char) + 4`. All Unicode characters above 255 map to `3` (`<unk>`).
- Encapsulates:
  - `encode(text, add_bos=False, add_eos=False)`: String to List of IDs.
  - `decode(ids)`: List of IDs to String.

### 2. Character EFLA Predictor Model (`EFLACharacterModel`)
A lightweight character-level sequence model wraps the canonical `EFLACell` (defined in `sef_gram/model.py`):
```python
import torch
import torch.nn as nn
from sef_gram.model import EFLACell

class EFLACharacterModel(nn.Module):
    def __init__(self, vocab_size=260, latent_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.head = nn.Linear(latent_dim, vocab_size)

    def forward(self, input_ids, s_prev=None):
        # input_ids: (B, SeqLen)
        # s_prev: (B, D, D), if None, initialized to zero/identity matrix
        B, T = input_ids.shape
        device = input_ids.device
        
        if s_prev is None:
            s_prev = torch.zeros(B, self.latent_dim, self.latent_dim, device=device)
            
        embeddings = self.embedding(input_ids) # (B, T, D)
        
        logits = []
        s_curr = s_prev
        z_curr = torch.zeros(B, self.latent_dim, device=device)
        
        for t in range(T):
            # Input to EFLA is character embedding e_t
            e_t = embeddings[:, t, :] # (B, D)
            # Transition memory and get next latent representation
            z_curr, s_curr = self.cell(e_t, s_curr)
            # Output projection to character vocabulary
            out = self.head(z_curr) # (B, vocab_size)
            logits.append(out.unsqueeze(1))
            
        logits = torch.cat(logits, dim=1) # (B, T, vocab_size)
        return logits, s_curr
```

### 3. Data Sources (`Synthetic` + `Hugging Face`)

#### Synthetic Terminal Generator
Produces sequences of the following template:
```bash
$ export VAR_0=VAL_0
$ export VAR_1=VAL_1
...
$ cat log.txt
[Distractor text of length L characters...]
$ echo $VAR_0
> VAL_0
```
- Distractor text is procedurally generated with real logs-like tokens to challenge linear attention.
- We test exact associative retrieval accuracy:
  - If predicted value matches `VAL_0` character-by-character, it's correct.
  - We systematically sweep over $L \in \{50, 100, 250, 500, 1000\}$ and $N \in \{1, 3, 5\}$ variables to map the capacity boundary.

#### Hugging Face Endless Terminals Dataset Streamer
- Streams JSON files and scripts from `obiwan96/endless-terminals`.
- Extracts instructions, bash solution command files (`solve.sh`), and task logs.
- Includes a sophisticated **Offline Fallback Engine** generating realistic git commits, docker builds, python exceptions, and bash loops when offline.

---

## Optimization Strategy
- We will optimize the model using our hardened `MuonWithAuxAdam` (Gram Newton-Schulz) optimizer.
- Matrix weights (`embedding.weight`, `cell.w_k.weight`, `cell.w_v.weight`, `head.weight`) are optimized via `Muon` with $lr=0.02$.
- 1D biases and trainable alpha scales are optimized via `AdamW` with $lr=3\times 10^{-4}$ and weight decay $0.01$.
- mixed-precision fp16 training is used for computational efficiency and Solomonoff induction regularization as per DEC-008.

---

## Verification Plan

### Automated Evaluators
1. **Perplexity / Cross-Entropy Evaluation**: Measure character-level cross-entropy loss (in nats/character) on held-out test data from the streaming terminal logs dataset.
2. **Associative Retrieval Capacity Sweeps**: Measure retrieval accuracy across varying distractor context lengths ($L$) and variable count ($N$) for different memory matrix sizes ($D \in \{16, 32, 64\}$) to identify the hard limits.

### Unit Tests
- Add a new test suite [`tests/test_tokenizer_and_benchmark.py`](file:///E:/experiments/SEF-GRAM/tests/test_tokenizer_and_benchmark.py) to verify:
  1. Tokenizer correctness (exact round-trip identity mapping, unknown token fallback).
  2. Data stream generator outputs.
  3. Character model forward pass stability.
