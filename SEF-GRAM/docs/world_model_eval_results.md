# World Model Evaluation Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

Current test suite passes locally:

```text
pytest tests/test_full_system.py tests/test_benchmarks.py tests/test_world_model.py tests/test_world_eval.py
13 passed
```

## Evaluation setup

Script:

```bash
python experiments/world_model_eval.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 5
```

The evaluation compares:

- `sef_gram`: latent SEF-GRAM world model with stochastic latent core and EFLA transition.
- `mlp_baseline`: direct one-step MLP baseline from `obs_t + action_t` to `next_obs_t, reward_t, done_t`.

## GPU run A

| model | one_step_total | one_step_obs_mse | one_step_reward_mse | open_loop_obs_energy |
|---|---:|---:|---:|---:|
| sef_gram | 0.1403 | 0.0082 | 0.0692 | 0.0725 |
| mlp_baseline | 0.1966 | 0.0046 | 0.1576 | 0.0840 |

Interpretation:

- SEF-GRAM has better total loss and reward prediction.
- MLP has lower one-step observation MSE.
- SEF-GRAM has slightly lower open-loop observation energy in this run.

## GPU run B with CSV export

Command:

```bash
python experiments/world_model_eval.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 5 --export-csv results_world_eval.csv
```

| model | one_step_total | one_step_obs_mse | one_step_reward_mse | open_loop_obs_energy |
|---|---:|---:|---:|---:|
| sef_gram | 0.1417 | 0.0077 | 0.0693 | 0.0793 |
| mlp_baseline | 0.2375 | 0.0041 | 0.1996 | 0.0751 |

Interpretation:

- SEF-GRAM again has better total loss and reward prediction.
- MLP again has lower one-step observation MSE.
- Open-loop proxy is close and not decisive.

## CPU smoke run

Command:

```bash
python experiments/world_model_eval.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 3 --device cpu --latent-dim 32 --hidden-dim 64
```

| model | one_step_total | one_step_obs_mse | one_step_reward_mse | open_loop_obs_energy |
|---|---:|---:|---:|---:|
| sef_gram | 0.7706 | 0.0775 | 0.1457 | 0.0281 |
| mlp_baseline | 0.9969 | 0.0523 | 0.8047 | 0.0027 |

Interpretation:

- CPU smoke runs are noisy and too short to support architecture-level conclusions.
- Even in the short run, SEF-GRAM predicts rewards better, while MLP predicts observations more conservatively.

## Current conclusion

The current result is mixed but useful:

1. SEF-GRAM is consistently better on total loss and reward prediction in the longer GPU runs.
2. MLP is consistently better on one-step observation MSE on these toy environments.
3. The toy environments are simple enough that a direct MLP can solve local transition prediction very efficiently.
4. The current open-loop proxy is not a true simulator rollout, because the toy generators are stateless samplers.

## Next work

1. Add true stateful rollout environments for GridWorld and LinePhysics.
2. Evaluate multi-step prediction against real simulator transitions, not only open-loop energy.
3. Add per-environment metrics instead of mixed-only metrics.
4. Add harder tasks where memory and latent planning should matter: delayed reward, hidden target, key-door gridworld, Sokoban-lite.
5. Keep MLP as the default baseline for every new environment.
