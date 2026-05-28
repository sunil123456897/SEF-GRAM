# Memory Key-Door GRU Baseline Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

The partially observable Memory Key-Door benchmark now compares three models:

- `sef_gram_memory`: recurrent latent/EFLA world model.
- `gru_memory`: recurrent GRU baseline.
- `mlp_memory`: direct MLP baseline without explicit recurrent state.

The task hides key/door/goal facts after the initial observation, so the model must carry hidden state across rollout steps.

## CPU smoke run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 20 --batch-size 16 --eval-batches 3 --rollout-horizon 5 --device cpu --latent-dim 32 --hidden-dim 64
```

Results:

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0825 | 0.0366 | 0.4497 | 53.75% | 78.33% | 98.75% | 95.00% |
| gru_memory | 0.1129 | 0.0220 | 0.6956 | 46.67% | 45.83% | 98.75% | 45.00% |
| mlp_memory | 0.1371 | 0.0011 | 0.6977 | 47.50% | 37.08% | 99.17% | 5.83% |

Interpretation:

- In the short CPU run, SEF-GRAM is clearly better on hidden-state observation rollout, blocked-door prediction, and done prediction.
- GRU is stronger than MLP on done accuracy but still far behind SEF-GRAM after only 20 steps.
- MLP reward MSE is misleadingly low because sparse/default rewards are easy to approximate without understanding events.

## GPU run

Command:

```bash
python experiments/memory_keydoor_compare.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Results:

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0067 | 0.0156 | 0.0724 | 95.42% | 98.48% | 99.24% | 98.51% |
| gru_memory | 0.0088 | 0.0165 | 0.0833 | 94.20% | 98.56% | 99.07% | 98.44% |
| mlp_memory | 0.0233 | 0.0157 | 0.0856 | 88.64% | 94.41% | 92.03% | 98.52% |

## Current conclusion

The GRU baseline is a much stronger opponent than MLP, and the result is still favorable to SEF-GRAM on the most important memory/world-model metrics:

1. SEF-GRAM has the best hidden-state observation rollout:
   - SEF-GRAM: 0.0067.
   - GRU: 0.0088.
   - MLP: 0.0233.
2. SEF-GRAM has the best done BCE:
   - SEF-GRAM: 0.0724.
   - GRU: 0.0833.
   - MLP: 0.0856.
3. SEF-GRAM has the best `has_key` and `key_pickup` accuracies:
   - has_key: 95.42% vs GRU 94.20% vs MLP 88.64%.
   - key_pickup: 99.24% vs GRU 99.07% vs MLP 92.03%.
4. GRU is slightly better on blocked-door accuracy:
   - GRU: 98.56%.
   - SEF-GRAM: 98.48%.
5. Done accuracy is essentially tied across all three after 500 steps, so done BCE is more informative than threshold accuracy here.

## Implication

This is the strongest architecture-level result so far. SEF-GRAM is not only beating a memoryless MLP; it also slightly outperforms a recurrent GRU baseline on aggregate hidden-state rollout, done calibration, and key causal-state tracking.

The result is still not final proof of superiority because:

- only one GPU seed is documented;
- the environment is synthetic and low-dimensional;
- the GRU baseline may improve with tuning;
- event rates/recalls should be inspected from CSV output, not only compact summary accuracy.

## Next work

1. Add multi-seed mean/std for `sef_gram_memory`, `gru_memory`, and `mlp_memory`.
2. Add compact printing of event recall and event rates.
3. Add harder settings:
   - lower probability of starting with `has_key=1`;
   - larger grid;
   - longer horizon;
   - rarer key/door/goal events.
4. Add planning evaluation: use the learned model to choose actions, not only predict dynamics.
5. Add LSTM or small Transformer recurrent baseline if needed.
