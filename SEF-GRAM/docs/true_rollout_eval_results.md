# True Multi-Step Rollout Evaluation Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

True rollout evaluation was added with stateful rollout environments:

- `GridWorldRolloutEnv`
- `LinePhysicsRolloutEnv`

The evaluation compares model predictions fed back into the next step against real environment transitions.

## CPU smoke run

Command:

```bash
python experiments/world_model_rollout_eval.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 3 --device cpu --latent-dim 32 --hidden-dim 64
```

Results:

| model | env | h1_obs_mse | h3_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram | gridworld | 0.4224 | 0.4079 | 0.4211 | 0.1204 |
| sef_gram | line_physics | 0.2922 | 0.2822 | 0.2802 | 0.5256 |
| mlp_baseline | gridworld | 0.2925 | 0.3093 | 0.3014 | 0.0507 |
| mlp_baseline | line_physics | 0.3085 | 0.3199 | 0.3128 | 0.6578 |

Interpretation:

- The short CPU run is too small to support strong conclusions.
- MLP is better on GridWorld in this smoke run.
- SEF-GRAM is slightly better than MLP on LinePhysics average observation error and reward error.

## GPU run

Command:

```bash
python experiments/world_model_rollout_eval.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Results:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram | gridworld | 0.0407 | 0.1335 | 0.1076 | 0.0733 |
| sef_gram | line_physics | 0.0370 | 0.2205 | 0.1383 | 0.2738 |
| mlp_baseline | gridworld | 0.0266 | 1.0475 | 0.2423 | 0.3640 |
| mlp_baseline | line_physics | 0.0137 | 75.6686 | 12.3460 | 19.5224 |

Interpretation:

- MLP has better one-step observation MSE on both environments.
- SEF-GRAM is much more stable over 10-step rollout.
- On GridWorld, SEF-GRAM has lower average rollout observation error and lower reward error.
- On LinePhysics, MLP explodes during self-fed rollout, while SEF-GRAM remains bounded enough to keep multi-step error much lower.

## Current conclusion

This is the strongest Phase 2 result so far:

1. MLP can learn very accurate one-step transitions on toy environments.
2. MLP predictions degrade badly when fed back recursively, especially on continuous dynamics.
3. SEF-GRAM has weaker one-step precision in some cases but better multi-step rollout stability.
4. This supports the motivation for latent world-model dynamics instead of only direct one-step regression.

## Caveats

- The environments are still toy tasks.
- The MLP baseline is not output-clamped or trained with multi-step rollout loss.
- The current SEF-GRAM world model is still one-step trained; rollout robustness may be improved by adding multi-step training loss.

## Next work

1. Add multi-step rollout loss during training.
2. Add output/domain constraints for all models for fair bounded rollout comparisons.
3. Add harder rollout environments: hidden target, delayed reward, key-door gridworld, Sokoban-lite.
4. Add multi-seed mean/std reporting.
5. Export true rollout evaluation to CSV by default for reproducibility.
