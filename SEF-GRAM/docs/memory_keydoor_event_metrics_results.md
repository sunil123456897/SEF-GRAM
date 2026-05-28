# Memory Key-Door Event Metrics Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

The partially observable key-door benchmark now reports both aggregate rollout losses and event-conditioned metrics.

Environment:

- At `t=0`, the model sees full facts: `agent`, `key`, `door`, `goal`, `has_key`.
- At later steps, `key`, `door`, and `goal` are hidden.
- The model must preserve hidden facts in memory to predict movement constraints and terminal events.

Compared models:

- `sef_gram_memory`: recurrent latent/EFLA state through `init_state()` and `stateful_step()`.
- `mlp_memory`: direct MLP with self-fed observations and no explicit recurrent memory.

## CPU smoke run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 5 --device cpu --latent-dim 32 --hidden-dim 64
```

Results:

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0910 | 0.0427 | 0.4348 | 41.67% | 73.75% | 98.75% | 96.67% |
| mlp_memory | 0.1162 | 0.0052 | 0.6974 | 48.75% | 30.42% | 99.58% | 7.08% |

Interpretation:

- The CPU smoke run is short and noisy, but already shows that SEF-GRAM handles blocked-door and done prediction much better.
- MLP has lower reward MSE because the reward distribution is sparse/default-heavy, so predicting near-default values is often sufficient.
- MLP's very low done accuracy indicates that aggregate reward MSE alone is misleading.

## GPU run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Results:

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0073 | 0.0173 | 0.0781 | 94.64% | 98.53% | 99.17% | 98.34% |
| mlp_memory | 0.0240 | 0.0161 | 0.0885 | 89.15% | 95.23% | 95.95% | 98.46% |

## Current conclusion

The GPU result supports the memory-world-model hypothesis:

1. SEF-GRAM is much better on hidden-state observation rollout:
   - obs MSE: 0.0073 vs 0.0240.
2. SEF-GRAM is better on causal memory variables:
   - has_key accuracy: 94.64% vs 89.15%.
   - blocked-door accuracy: 98.53% vs 95.23%.
   - key-pickup accuracy: 99.17% vs 95.95%.
3. Done accuracy is essentially tied:
   - SEF-GRAM: 98.34%.
   - MLP: 98.46%.
4. Reward MSE is nearly tied, with MLP slightly lower:
   - MLP: 0.0161.
   - SEF-GRAM: 0.0173.

The important result is not raw reward MSE, but the combination of lower observation error and higher event accuracy on hidden causal variables.

## Caveats

- Only one GPU seed is documented.
- The MLP baseline has no recurrent memory; a stronger recurrent baseline such as GRU/LSTM is still needed.
- Event rates are not yet printed in the compact summary, although they are included in the returned row/CSV.
- Accuracy can be inflated for rare events; recall and event-conditioned precision should be examined in CSV runs.

## Next work

1. Add compact printing of event rates and recalls:
   - `blocked_recall`, `key_pickup_recall`, `done_recall`.
2. Add GRU/LSTM baseline.
3. Add multi-seed mean/std evaluation.
4. Add planning evaluation where the agent must choose actions to collect key and reach goal.
5. Increase difficulty by lowering initial `has_key=1` probability and increasing horizon.
