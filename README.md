# SEF-GRAM

Integrated research MVP for a stochastic latent reasoning/world-model agent.

## Status

Phase 2 is closed as a successful toy-world-model MVP.

SEF-GRAM now includes:

- stochastic latent recursive core;
- EFLA-style recurrent memory transition;
- universal `WorldBatch` world-model API;
- true multi-step rollout evaluation;
- multi-step rollout training;
- MLP and GRU baselines;
- partially observable Memory Key-Door benchmark;
- multi-seed result aggregation;
- first model-based planning evaluation.

Main conclusion:

> SEF-GRAM is consistently useful on partially observable and long-horizon toy world-model tasks. It clearly beats MLP and slightly beats GRU on hidden-state rollout, causal-state tracking, done calibration, and planning success in the current Memory Key-Door benchmark.

This is not a final proof of a universal architecture. It is a strong Phase 2 milestone. Phase 3 should focus on planning/value learning and harder environments.

## Core

- `SEF-GRAM/sef_gram/full_system.py` — integrated VJEPA-style latent encoder, EFLA memory transition, GRAM-style stochastic recursive rollout, GDPO/ECHO objectives, and PoE planner.
- `SEF-GRAM/sef_gram/world_model.py` — universal SEF-GRAM latent world model and recurrent state API.
- `SEF-GRAM/sef_gram/world_baselines.py` — MLP and GRU world-model baselines.
- `SEF-GRAM/sef_gram/world_envs.py` — mixed one-step world-model training environments.
- `SEF-GRAM/sef_gram/world_rollout_envs.py` — stateful rollout environments.

## Key experiments

- `SEF-GRAM/experiments/world_model_train.py` — basic world-model training.
- `SEF-GRAM/experiments/world_model_eval.py` — one-step evaluation against MLP.
- `SEF-GRAM/experiments/world_model_rollout_eval.py` — true multi-step rollout evaluation.
- `SEF-GRAM/experiments/world_model_multistep_train.py` — SEF-GRAM multi-step rollout training.
- `SEF-GRAM/experiments/world_model_multistep_compare.py` — fair SEF-GRAM vs MLP multi-step comparison.
- `SEF-GRAM/experiments/memory_keydoor_compare.py` — partially observable memory benchmark with SEF-GRAM, GRU, and MLP.
- `SEF-GRAM/experiments/memory_keydoor_multiseed.py` — multi-seed aggregation for Memory Key-Door.
- `SEF-GRAM/experiments/memory_keydoor_planning_eval.py` — first model-based planning evaluation.

## Phase 2 headline results

### Multi-seed Memory Key-Door

Three GPU seeds: 37, 38, 39.

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc |
|---|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0068 ± 0.0001 | 0.0168 ± 0.0012 | 0.0755 ± 0.0030 | 95.05% ± 0.32% | 98.44% ± 0.16% |
| gru_memory | 0.0084 ± 0.0004 | 0.0156 ± 0.0013 | 0.0792 ± 0.0057 | 94.84% ± 0.56% | 98.47% ± 0.14% |
| mlp_memory | 0.0211 ± 0.0024 | 0.0161 ± 0.0014 | 0.0901 ± 0.0064 | 92.24% ± 3.21% | 95.90% ± 1.30% |

### Model-based planning

Geometry-aware planning, one GPU run:

| policy/model | success | key_rate | avg_reward | avg_blocked |
|---|---:|---:|---:|---:|
| random_policy | 5.23% | 27.46% | -0.2719 | 0.4352 |
| oracle_candidate | 94.49% | 97.19% | 3.1937 | 0.0465 |
| sef_gram_memory | 23.32% | 86.88% | 0.3971 | 0.3148 |
| gru_memory | 16.88% | 85.98% | 0.2077 | 0.3473 |
| mlp_memory | 12.62% | 28.32% | -0.0078 | 0.4125 |

Interpretation:

- SEF-GRAM is useful for action selection, not only prediction.
- SEF-GRAM beats random, GRU, and MLP in the current planning setup.
- The large gap to `oracle_candidate` shows that planning remains the weakest part of the stack.

See the full report:

- `SEF-GRAM/docs/phase2_final_report.md`

## Quick checks

Run from the inner project directory:

```bash
pytest tests/test_full_system.py tests/test_benchmarks.py tests/test_world_model.py tests/test_world_eval.py tests/test_world_rollout_eval.py tests/test_world_multistep_train.py tests/test_world_multistep_compare.py tests/test_memory_keydoor.py tests/test_memory_keydoor_multiseed.py tests/test_memory_keydoor_planning.py
```

## Reproduce strongest results

Multi-seed Memory Key-Door:

```bash
python experiments/memory_keydoor_multiseed.py --seeds 37,38,39 --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10 --export-csv results_memory_keydoor_multiseed.csv --export-raw-csv results_memory_keydoor_multiseed_raw.csv
```

Planning evaluation:

```bash
python experiments/memory_keydoor_planning_eval.py --steps 500 --batch-size 128 --eval-batches 20 --plan-horizon 20 --num-candidates 256 --export-csv results_memory_keydoor_planning.csv
```

## Legacy benchmarks

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

## Phase 3 direction

Phase 3 should continue in a separate workstream.

Recommended next scope:

1. learned value/planning head trained from rollouts;
2. CEM/MPC candidate optimization;
3. no-op/stop action to reduce random padding artifacts;
4. multi-seed planning evaluation;
5. LSTM and small Transformer recurrent baselines;
6. harder environments after planning becomes stable.
