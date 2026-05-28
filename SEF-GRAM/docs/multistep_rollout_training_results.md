# Multi-Step Rollout Training Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

Multi-step rollout loss was added through:

- `experiments/world_model_multistep_train.py`
- `tests/test_world_multistep_train.py`

The objective combines one-step world-model loss with differentiable self-fed rollout loss over stateful rollout environments.

## CPU smoke run

Command:

```bash
python experiments/world_model_multistep_train.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 3 --device cpu --latent-dim 32 --hidden-dim 64
```

Training summary:

| step | total | one_step | rollout | rollout_obs | rollout_reward |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.5183 | 1.1512 | 2.3671 | 0.3072 | 1.8975 |
| 20 | 1.7888 | 0.9220 | 0.8668 | 0.3111 | 0.4092 |

Evaluation:

| model | env | h1_obs_mse | h3_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.3427 | 0.3158 | 0.3317 | 0.3606 |
| sef_gram_multistep | line_physics | 0.2738 | 0.3002 | 0.2865 | 0.4727 |

Interpretation:

- The CPU smoke run verifies the multi-step objective runs and optimizes without crashing.
- The training rollout loss drops from 2.3671 to 0.8668 in 20 steps.
- The run is too short for architecture-level conclusions.

## GPU multi-step run

Command:

```bash
python experiments/world_model_multistep_train.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Training summary:

| step | total | one_step | rollout | rollout_obs | rollout_reward |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.2535 | 0.9905 | 1.2630 | 0.3389 | 0.8151 |
| 100 | 0.6772 | 0.3094 | 0.3678 | 0.1559 | 0.1792 |
| 200 | 0.4275 | 0.1351 | 0.2924 | 0.1196 | 0.1254 |
| 300 | 0.4566 | 0.1926 | 0.2640 | 0.0999 | 0.1220 |
| 400 | 0.3646 | 0.1624 | 0.2022 | 0.0713 | 0.1004 |
| 500 | 0.3491 | 0.1461 | 0.2030 | 0.0611 | 0.1024 |

Evaluation:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.0377 | 0.0925 | 0.0739 | 0.0499 |
| sef_gram_multistep | line_physics | 0.0156 | 0.0751 | 0.0461 | 0.1176 |

## Comparison against previous one-step training

Previous one-step-only SEF-GRAM run:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram | gridworld | 0.0392 | 0.1359 | 0.1082 | 0.0604 |
| sef_gram | line_physics | 0.0374 | 0.2533 | 0.1540 | 0.2465 |

MLP baseline from the same comparison run:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| mlp_baseline | gridworld | 0.0282 | 1.6048 | 0.3402 | 0.6282 |
| mlp_baseline | line_physics | 0.0137 | 90.0836 | 14.8522 | 27.6498 |

## Current conclusion

Multi-step rollout training produces a clear improvement over one-step-only SEF-GRAM:

- GridWorld h10 observation MSE: 0.1359 -> 0.0925.
- GridWorld average observation MSE: 0.1082 -> 0.0739.
- GridWorld reward MSE: 0.0604 -> 0.0499.
- LinePhysics h10 observation MSE: 0.2533 -> 0.0751.
- LinePhysics average observation MSE: 0.1540 -> 0.0461.
- LinePhysics reward MSE: 0.2465 -> 0.1176.

This is the strongest evidence so far that SEF-GRAM should be trained as a world model with recursive rollout loss rather than only one-step prediction.

## Caveats

- Still only toy environments.
- Only one seed is documented here.
- Baseline MLP is not trained with equivalent multi-step rollout loss yet.
- The current rollout loss is simple MSE/BCE and does not yet use uncertainty-aware or constraint-aware objectives.

## Next work

1. Add multi-step MLP baseline training for fairness.
2. Add multi-seed mean/std reporting.
3. Add harder memory/planning environments such as hidden target, key-door gridworld, and Sokoban-lite.
4. Add automatic CSV output for multi-step training and evaluation.
5. Add action planning using the trained world model instead of only evaluating prediction quality.
