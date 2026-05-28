# Partially Observable Key-Door Memory Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

This benchmark tests whether a model can carry hidden world facts through recurrent state.

Environment:

- At `t=0`, the model observes full facts: `agent`, `key`, `door`, `goal`, `has_key`.
- At later steps, `key`, `door`, and `goal` are masked to zero.
- The model must remember hidden facts to predict movement constraints, key pickup, reward, and done.

Compared models:

- `sef_gram_memory`: uses `UniversalWorldModel.init_state()` and `stateful_step()` with recurrent latent/EFLA memory.
- `mlp_memory`: direct MLP baseline with self-fed observations and no explicit recurrent memory.

## CPU smoke run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 5 --device cpu --latent-dim 32 --hidden-dim 64
```

Training summary:

| model | final_train_total | final_train_obs | final_train_reward | final_train_done |
|---|---:|---:|---:|---:|
| sef_gram_memory | 0.2123 | 0.1052 | 0.0156 | 0.4574 |
| mlp_memory | 0.2775 | 0.1363 | 0.0015 | 0.6986 |

Evaluation:

| model | rollout_obs_mse_avg | rollout_reward_mse_avg | rollout_done_bce_avg |
|---|---:|---:|---:|
| sef_gram_memory | 0.0910 | 0.0427 | 0.4348 |
| mlp_memory | 0.1162 | 0.0052 | 0.6974 |

Interpretation:

- On the short CPU run, SEF-GRAM has better observation rollout and done prediction.
- MLP has better reward MSE, likely because rewards are sparse and mostly near the default step penalty.

## GPU run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Training summary:

| model | final_train_total | final_train_obs | final_train_reward | final_train_done |
|---|---:|---:|---:|---:|
| sef_gram_memory | 0.0315 | 0.0063 | 0.0131 | 0.0606 |
| mlp_memory | 0.0612 | 0.0253 | 0.0177 | 0.0908 |

Evaluation:

| model | rollout_obs_mse_avg | rollout_reward_mse_avg | rollout_done_bce_avg |
|---|---:|---:|---:|
| sef_gram_memory | 0.0073 | 0.0173 | 0.0781 |
| mlp_memory | 0.0240 | 0.0161 | 0.0885 |

## Current conclusion

This is the strongest memory-specific result so far:

1. SEF-GRAM clearly beats MLP on hidden-state observation rollout:
   - obs MSE: 0.0073 vs 0.0240.
2. SEF-GRAM also improves done prediction:
   - done BCE: 0.0781 vs 0.0885.
3. Reward MSE is nearly tied, with MLP slightly lower:
   - reward MSE: 0.0161 vs SEF-GRAM 0.0173.
4. The benchmark validates the need for recurrent latent state on partially observable world-model tasks.

## Caveats

- The benchmark still uses synthetic toy dynamics.
- Reward is sparse/default-heavy, so reward MSE can be misleading without event-conditioned metrics.
- Only one GPU seed is documented.
- The MLP baseline has no recurrent memory; a fairer stronger baseline would be GRU/LSTM.

## Next work

1. Add event-conditioned metrics:
   - key pickup accuracy;
   - blocked-door prediction accuracy;
   - done accuracy;
   - has_key accuracy.
2. Add a GRU/LSTM baseline for a stronger memory comparison.
3. Add multi-seed mean/std reporting.
4. Add planning evaluation: choose actions using the learned world model to collect key and reach goal.
5. Increase task difficulty with longer horizons and lower probability of starting with `has_key=1`.
