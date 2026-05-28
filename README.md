# SEF-GRAM

Integrated research MVP for a stochastic latent reasoning agent.

## Core

- `SEF-GRAM/sef_gram/full_system.py` — integrated VJEPA-style latent encoder, EFLA memory transition, GRAM-style stochastic recursive rollout, GDPO/ECHO objectives, and PoE planner.
- `SEF-GRAM/experiments/full_nqueens_benchmark.py` — multi-trajectory N-Queens benchmark.
- `SEF-GRAM/experiments/terminal_echo_benchmark.py` — terminal-memory ECHO benchmark.

## Quick checks

Run from the inner project directory:

```bash
pytest tests/test_full_system.py tests/test_benchmarks.py
python examples/run_full_system_smoke.py
```

## Benchmarks

Fast CPU smoke runs:

```bash
python experiments/full_nqueens_benchmark.py --board-size 4 --train-steps 20 --batch-size 8 --num-trajectories 8 --device cpu
python experiments/terminal_echo_benchmark.py --train-steps 20 --batch-size 4 --distractor-len 32 --eval-cases 5 --device cpu
```

Longer runs:

```bash
python experiments/full_nqueens_benchmark.py --board-size 8 --train-steps 500 --batch-size 64 --num-trajectories 16
python experiments/terminal_echo_benchmark.py --train-steps 600 --batch-size 32 --distractor-len 160
```
