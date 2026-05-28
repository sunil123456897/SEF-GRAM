# SEF-GRAM Phase 2 Final Report

Owner: solo research project
Status: Phase 2 closed; Phase 3 should continue in a separate planning/action-learning track.
Last updated: 2026-05-28
Expires/recheck by: 2026-06-28

## Executive summary

Phase 2 successfully moved SEF-GRAM from a small stochastic latent reasoning MVP into a working toy universal world-model research stack.

The project now has:

- a common `WorldBatch` API;
- mixed environment training;
- a universal latent world model;
- direct MLP, recurrent GRU, and SEF-GRAM memory baselines;
- true multi-step rollout evaluation;
- multi-step rollout training;
- partially observable memory benchmarks;
- multi-seed benchmark aggregation;
- first model-based planning evaluation.

The strongest result is not that SEF-GRAM wins every metric. The strongest result is narrower and more defensible:

> SEF-GRAM is consistently useful on partially observable and long-horizon world-model tasks. It beats MLP clearly and slightly beats GRU on hidden-state rollout, causal-state tracking, done calibration, and model-based planning success in the current toy Memory Key-Door benchmark.

## What was implemented

### Core world-model layer

Files:

- `sef_gram/world_model.py`
- `sef_gram/world_envs.py`
- `sef_gram/world_rollout_envs.py`
- `sef_gram/world_baselines.py`

Implemented:

- `UniversalWorldModel`
- `WorldBatch`
- `pad_obs`
- `MLPWorldModel`
- `GRUWorldModel`
- stateful recurrent SEF-GRAM API:
  - `init_state(obs)`
  - `stateful_step(state, action)`

### Environments

Implemented toy environments:

- `GridWorld`
- `KeyDoorGridWorld`
- `LinePhysics`
- `MemoryKeyDoorSequenceEnv`

The most important environment is `MemoryKeyDoorSequenceEnv`:

- at `t=0`, the model sees full facts: agent, key, door, goal, has_key;
- after `t=0`, key/door/goal are hidden;
- the model must carry hidden facts in memory to predict key pickup, blocked movement, reward, and done.

### Evaluation and training scripts

Files:

- `experiments/world_model_train.py`
- `experiments/world_model_eval.py`
- `experiments/world_model_rollout_eval.py`
- `experiments/world_model_multistep_train.py`
- `experiments/world_model_multistep_compare.py`
- `experiments/memory_keydoor_compare.py`
- `experiments/memory_keydoor_multiseed.py`
- `experiments/memory_keydoor_planning_eval.py`

### Documentation added during Phase 2

Important result docs:

- `docs/world_model_eval_results.md`
- `docs/true_rollout_eval_results.md`
- `docs/multistep_rollout_training_results.md`
- `docs/fair_multistep_comparison_results.md`
- `docs/key_door_multistep_results.md`
- `docs/memory_keydoor_results.md`
- `docs/memory_keydoor_event_metrics_results.md`
- `docs/memory_keydoor_gru_baseline_results.md`
- `docs/memory_keydoor_multiseed_results.md`
- `docs/memory_keydoor_planning_results.md`
- `docs/phase2_final_report.md`

## Key results

### 1. True multi-step rollout: SEF-GRAM beats one-step MLP stability

GPU run, horizon 10:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram | gridworld | 0.0407 | 0.1335 | 0.1076 | 0.0733 |
| sef_gram | line_physics | 0.0370 | 0.2205 | 0.1383 | 0.2738 |
| mlp_baseline | gridworld | 0.0266 | 1.0475 | 0.2423 | 0.3640 |
| mlp_baseline | line_physics | 0.0137 | 75.6686 | 12.3460 | 19.5224 |

Interpretation:

- MLP can be better at one-step prediction.
- SEF-GRAM is much more stable under recursive self-fed rollout, especially on continuous dynamics.

### 2. Multi-step training improves SEF-GRAM rollout quality

One-step-only SEF-GRAM vs multi-step-trained SEF-GRAM:

| model | env | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|
| one-step SEF-GRAM | gridworld | 0.1359 | 0.1082 | 0.0604 |
| multi-step SEF-GRAM | gridworld | 0.0925 | 0.0739 | 0.0499 |
| one-step SEF-GRAM | line_physics | 0.2533 | 0.1540 | 0.2465 |
| multi-step SEF-GRAM | line_physics | 0.0751 | 0.0461 | 0.1176 |

Interpretation:

- Training on recursive rollout loss is necessary.
- It especially improves continuous long-horizon dynamics.

### 3. Fair multi-step comparison: SEF-GRAM vs MLP

Both models trained with the same multi-step objective:

| model | env | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.0970 | 0.0792 | 0.0526 |
| mlp_multistep | gridworld | 0.0913 | 0.0724 | 0.0669 |
| sef_gram_multistep | line_physics | 0.0856 | 0.0582 | 0.1183 |
| mlp_multistep | line_physics | 0.1340 | 0.0650 | 0.2042 |

Interpretation:

- MLP remains very strong on simple fully observed grid transitions.
- SEF-GRAM is stronger on continuous long-horizon dynamics and reward prediction.

### 4. Memory Key-Door with GRU baseline

GPU run, partially observable Memory Key-Door:

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0067 | 0.0156 | 0.0724 | 95.42% | 98.48% | 99.24% | 98.51% |
| gru_memory | 0.0088 | 0.0165 | 0.0833 | 94.20% | 98.56% | 99.07% | 98.44% |
| mlp_memory | 0.0233 | 0.0157 | 0.0856 | 88.64% | 94.41% | 92.03% | 98.52% |

Interpretation:

- MLP is clearly weaker on hidden-state rollout and causal-state tracking.
- GRU is a strong baseline.
- SEF-GRAM slightly outperforms GRU on the most important aggregate memory/world-model metrics.

### 5. Multi-seed Memory Key-Door

Three GPU seeds: 37, 38, 39.

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc |
|---|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0068 ± 0.0001 | 0.0168 ± 0.0012 | 0.0755 ± 0.0030 | 95.05% ± 0.32% | 98.44% ± 0.16% |
| gru_memory | 0.0084 ± 0.0004 | 0.0156 ± 0.0013 | 0.0792 ± 0.0057 | 94.84% ± 0.56% | 98.47% ± 0.14% |
| mlp_memory | 0.0211 ± 0.0024 | 0.0161 ± 0.0014 | 0.0901 ± 0.0064 | 92.24% ± 3.21% | 95.90% ± 1.30% |

Interpretation:

- The SEF-GRAM advantage over GRU is small but stable on observation rollout, done calibration, and has_key tracking.
- MLP is clearly weaker and less stable.
- GRU is marginally better on blocked-door accuracy, but the difference is negligible.

### 6. First model-based planning result

Geometry-aware model-based planning, GPU run:

| policy/model | success | key_rate | avg_reward | avg_blocked |
|---|---:|---:|---:|---:|
| random_policy | 5.23% | 27.46% | -0.2719 | 0.4352 |
| oracle_candidate | 94.49% | 97.19% | 3.1937 | 0.0465 |
| sef_gram_memory | 23.32% | 86.88% | 0.3971 | 0.3148 |
| gru_memory | 16.88% | 85.98% | 0.2077 | 0.3473 |
| mlp_memory | 12.62% | 28.32% | -0.0078 | 0.4125 |

Interpretation:

- SEF-GRAM is useful for action selection, not only prediction.
- It beats random, GRU, and MLP in this planning setup.
- The gap to oracle is large, so planning is still the weakest part of the stack.

## Honest conclusion

Phase 2 is a success, but not a final proof of a universal architecture.

Supported claims:

1. SEF-GRAM can act as a trainable universal toy world-model core.
2. SEF-GRAM is better than MLP on long-horizon rollout and partial observability.
3. SEF-GRAM slightly outperforms GRU on the current Memory Key-Door benchmark across several key metrics.
4. SEF-GRAM can improve model-based planning outcomes over random, GRU, and MLP in the current setup.

Unsupported or not-yet-proven claims:

1. SEF-GRAM is not proven superior on all world-model tasks.
2. SEF-GRAM is not proven on high-dimensional visual environments.
3. SEF-GRAM is not proven against tuned Transformer/LSTM baselines.
4. SEF-GRAM is not yet a complete agent architecture.
5. Planning is not solved; there is a large gap to oracle candidate selection.

## Phase 2 final status

Recommended status:

> Phase 2 closed as a successful toy-world-model MVP. Continue Phase 3 in a new workstream focused on planning/value learning and harder environments.

## Repro commands

Run tests:

```bash
pytest tests/test_full_system.py tests/test_benchmarks.py tests/test_world_model.py tests/test_world_eval.py tests/test_world_rollout_eval.py tests/test_world_multistep_train.py tests/test_world_multistep_compare.py tests/test_memory_keydoor.py tests/test_memory_keydoor_multiseed.py tests/test_memory_keydoor_planning.py
```

Run multi-seed Memory Key-Door:

```bash
python experiments/memory_keydoor_multiseed.py --seeds 37,38,39 --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10 --export-csv results_memory_keydoor_multiseed.csv --export-raw-csv results_memory_keydoor_multiseed_raw.csv
```

Run planning evaluation:

```bash
python experiments/memory_keydoor_planning_eval.py --steps 500 --batch-size 128 --eval-batches 20 --plan-horizon 20 --num-candidates 256 --export-csv results_memory_keydoor_planning.csv
```

## Phase 3 handoff

Do not continue by adding more prediction metrics first. The next stage should focus on planning.

Recommended Phase 3 scope:

1. Add learned value/planning head trained from simulated rollouts.
2. Add CEM/MPC candidate optimization instead of mostly random shooting.
3. Add no-op/stop action to reduce padding artifacts.
4. Add multi-seed planning evaluation.
5. Add stronger recurrent baselines: LSTM and small Transformer.
6. Add harder environments only after planning is stable.

Phase 3 success criterion:

> SEF-GRAM should close a significant part of the gap between current learned planning and `oracle_candidate`, while maintaining or improving its advantage over GRU/MLP.
