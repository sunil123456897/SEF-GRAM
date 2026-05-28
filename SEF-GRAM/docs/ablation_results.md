# SEF-GRAM Ablation Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

The current implementation has passed the smoke and benchmark tests locally:

```text
pytest tests/test_full_system.py tests/test_benchmarks.py
7 passed
```

The results below are from local user runs after the integrated full-system and ablation suite were added.

## 1. N-Queens stochastic trajectory ablation

### CPU smoke run

Command:

```bash
python experiments/ablation_suite.py --device cpu --board-size 4 --nqueens-steps 30 --nqueens-batch-size 16 --nqueens-eval-trials 5 --nqueens-k-values 1,4,8 --terminal-steps 30 --terminal-batch-size 8 --terminal-distractor-len 32 --terminal-eval-cases 10 --latent-dim 32 --hidden-dim 64
```

Results:

| K | valid_solution_rate | unique_solution_count | trajectory_diversity | best_of_k_success |
|---:|---:|---:|---:|---:|
| 1 | 20.00% | 0.20 | 100.00% | 20.00% |
| 4 | 15.00% | 0.60 | 95.00% | 60.00% |
| 8 | 7.50% | 0.60 | 92.50% | 40.00% |

Interpretation:

- K=4 clearly improves best-of-K success compared with K=1 on the small CPU run.
- K=8 did not monotonically improve success in this short run, which suggests either undertraining or noisy evaluation.
- Diversity remains high, so the model is exploring multiple trajectories rather than collapsing to one sequence.

### GPU run

Command:

```bash
python experiments/ablation_suite.py --board-size 8 --nqueens-steps 300 --nqueens-batch-size 64 --nqueens-eval-trials 20 --nqueens-k-values 1,4,16 --terminal-steps 200 --terminal-batch-size 32 --terminal-distractor-len 80 --terminal-eval-cases 50
```

Results:

| K | valid_solution_rate | unique_solution_count | trajectory_diversity | best_of_k_success |
|---:|---:|---:|---:|---:|
| 1 | 10.00% | 0.10 | 100.00% | 10.00% |
| 4 | 18.75% | 0.70 | 97.50% | 60.00% |
| 16 | 11.88% | 1.75 | 97.19% | 75.00% |

Interpretation:

- The main GRAM-style hypothesis is supported in this run: more sampled trajectories improve best-of-K success and unique valid solution coverage.
- K=16 finds more unique valid solutions than K=4 and K=1.
- valid_solution_rate is not strictly monotonic, but best_of_k_success and unique_solution_count are the more relevant metrics for multi-trajectory search.

## 2. Terminal retrieval ablation

### CPU smoke run

Results:

| mode | exact_retrieval_accuracy | char_accuracy |
|---|---:|---:|
| neural_only | 0.00% | 12.50% |
| hybrid_kv_parser | 100.00% | 100.00% |

### GPU run

Results:

| mode | exact_retrieval_accuracy | char_accuracy |
|---|---:|---:|
| neural_only | 0.00% | 11.00% |
| hybrid_kv_parser | 100.00% | 100.00% |

Interpretation:

- The neural-only EFLA retrieval baseline fails on exact terminal variable lookup under the current architecture and training regime.
- The hybrid KV parser solves the terminal lookup task exactly.
- This means terminal key-value retrieval should be treated as an agent memory/interface problem, not as a pure latent-RNN memorization problem.

## Current conclusions

1. The strongest current result is the N-Queens K-ablation: stochastic recursive multi-trajectory sampling improves best-of-K success and solution coverage.
2. The terminal benchmark shows that deterministic tool-like memory is necessary for reliable exact shell-variable retrieval.
3. The project should now focus on strengthening the N-Queens/constraint-solving benchmark before adding more architecture complexity.

## Next work

1. Add multi-seed evaluation and aggregate mean/std metrics for N-Queens.
2. Add an explicit baseline comparison: single-trajectory model, stochastic multi-trajectory model, and simple MLP/Transformer policy.
3. Add JSON/CSV result export for reproducible experiment tracking.
4. Improve N-Queens training objective with diversity regularization or solution-coverage reward.
5. Keep terminal retrieval as hybrid-agent memory, not as proof of neural long-context recall.
