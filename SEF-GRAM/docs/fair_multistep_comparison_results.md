# Fair Multi-Step Comparison Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

This document records the first fair comparison where both SEF-GRAM and the MLP baseline are trained with the same multi-step rollout objective.

Compared models:

- `sef_gram_multistep`
- `mlp_multistep`

Both use:

```text
total = one_step_loss_weight * one_step_loss
      + rollout_loss_weight * multi_step_rollout_loss
```

## CPU smoke run

Command:

```bash
python experiments/world_model_multistep_compare.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 3 --device cpu --latent-dim 32 --hidden-dim 64
```

Results:

| model | env | h1_obs_mse | hN_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.3107 | 0.2871 | 0.2958 | 0.3308 |
| sef_gram_multistep | line_physics | 0.2292 | 0.2389 | 0.2324 | 0.3050 |
| mlp_multistep | gridworld | 0.3563 | 0.3753 | 0.3659 | 0.0413 |
| mlp_multistep | line_physics | 0.2833 | 0.2864 | 0.2849 | 0.6675 |

Interpretation:

- On the short CPU smoke run, SEF-GRAM has better rollout observation error on both environments.
- MLP has better GridWorld reward prediction, but worse LinePhysics reward prediction.
- The run is too short for final conclusions.

## GPU fair comparison run

Command:

```bash
python experiments/world_model_multistep_compare.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Results:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.0464 | 0.0970 | 0.0792 | 0.0526 |
| sef_gram_multistep | line_physics | 0.0207 | 0.0856 | 0.0582 | 0.1183 |
| mlp_multistep | gridworld | 0.0365 | 0.0913 | 0.0724 | 0.0669 |
| mlp_multistep | line_physics | 0.0176 | 0.1340 | 0.0650 | 0.2042 |

## Current conclusion

The fair comparison is mixed but positive for the SEF-GRAM direction:

1. On GridWorld, MLP is slightly better on observation rollout error:
   - h10 obs: 0.0913 vs SEF-GRAM 0.0970.
   - avg obs: 0.0724 vs SEF-GRAM 0.0792.
2. On GridWorld, SEF-GRAM is better on reward prediction:
   - reward MSE: 0.0526 vs MLP 0.0669.
3. On LinePhysics, SEF-GRAM is better on long-horizon observation stability:
   - h10 obs: 0.0856 vs MLP 0.1340.
   - avg obs: 0.0582 vs MLP 0.0650.
4. On LinePhysics, SEF-GRAM is also better on reward prediction:
   - reward MSE: 0.1183 vs MLP 0.2042.

This means the strongest current evidence for SEF-GRAM is not simple spatial one-step prediction, but continuous/multi-step dynamics and reward prediction.

## Caveats

- Only one GPU seed is documented here.
- The tasks are still simple toy environments.
- MLP is a strong baseline on low-dimensional deterministic environments.
- SEF-GRAM needs harder environments where latent state, memory, uncertainty, and multi-trajectory planning matter more.

## Next work

1. Add multi-seed mean/std for the fair comparison.
2. Add harder memory/planning environments:
   - hidden-target gridworld;
   - key-door gridworld;
   - delayed reward line physics;
   - Sokoban-lite.
3. Add action-planning evaluation using the learned world model.
4. Add CSV export to all benchmark scripts by default.
5. Add a small Transformer baseline for sequence/world-model comparison.
