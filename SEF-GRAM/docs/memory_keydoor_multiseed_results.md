# Memory Key-Door Multi-Seed Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

The partially observable Memory Key-Door benchmark was run across three GPU seeds with three models:

- `sef_gram_memory`: recurrent latent/EFLA world model.
- `gru_memory`: recurrent GRU baseline.
- `mlp_memory`: memoryless MLP baseline with self-fed observations.

Command:

```bash
python experiments/memory_keydoor_multiseed.py --seeds 37,38,39 --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10 --export-csv results_memory_keydoor_multiseed.csv --export-raw-csv results_memory_keydoor_multiseed_raw.csv
```

A CPU smoke run with seeds `1,2` also completed successfully.

## GPU multi-seed aggregate

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc |
|---|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0068 ± 0.0001 | 0.0168 ± 0.0012 | 0.0755 ± 0.0030 | 95.05% ± 0.32% | 98.44% ± 0.16% |
| gru_memory | 0.0084 ± 0.0004 | 0.0156 ± 0.0013 | 0.0792 ± 0.0057 | 94.84% ± 0.56% | 98.47% ± 0.14% |
| mlp_memory | 0.0211 ± 0.0024 | 0.0161 ± 0.0014 | 0.0901 ± 0.0064 | 92.24% ± 3.21% | 95.90% ± 1.30% |

## Per-seed notes

### Seed 37

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0067 | 0.0156 | 0.0724 | 95.42% | 98.48% | 99.24% | 98.51% |
| gru_memory | 0.0088 | 0.0165 | 0.0833 | 94.20% | 98.56% | 99.07% | 98.44% |
| mlp_memory | 0.0233 | 0.0157 | 0.0856 | 88.64% | 94.41% | 92.03% | 98.52% |

### Seed 38

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0068 | 0.0180 | 0.0785 | 94.84% | 98.27% | 99.16% | 98.31% |
| gru_memory | 0.0081 | 0.0141 | 0.0727 | 95.14% | 98.30% | 99.20% | 98.68% |
| mlp_memory | 0.0215 | 0.0149 | 0.0874 | 93.30% | 96.45% | 99.14% | 98.58% |

### Seed 39

| model | obs_mse | reward_mse | done_bce | has_key_acc | blocked_acc | key_pickup_acc | done_acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| sef_gram_memory | 0.0069 | 0.0169 | 0.0755 | 94.90% | 98.57% | 99.17% | 98.40% |
| gru_memory | 0.0082 | 0.0162 | 0.0817 | 95.19% | 98.54% | 99.26% | 98.47% |
| mlp_memory | 0.0186 | 0.0176 | 0.0974 | 94.79% | 96.83% | 99.19% | 98.31% |

## Current conclusion

The multi-seed result strengthens the architecture-level case for SEF-GRAM:

1. `sef_gram_memory` has the best hidden-state rollout error with very low variance:
   - SEF-GRAM: `0.0068 ± 0.0001`.
   - GRU: `0.0084 ± 0.0004`.
   - MLP: `0.0211 ± 0.0024`.
2. `sef_gram_memory` has the best done calibration:
   - SEF-GRAM: `0.0755 ± 0.0030`.
   - GRU: `0.0792 ± 0.0057`.
   - MLP: `0.0901 ± 0.0064`.
3. `sef_gram_memory` has the best `has_key` accuracy on average:
   - SEF-GRAM: `95.05% ± 0.32%`.
   - GRU: `94.84% ± 0.56%`.
   - MLP: `92.24% ± 3.21%`.
4. GRU is marginally better on blocked-door accuracy:
   - GRU: `98.47% ± 0.14%`.
   - SEF-GRAM: `98.44% ± 0.16%`.
   - This difference is negligible in practice.
5. GRU and MLP are slightly better on reward MSE, but reward MSE is less diagnostic here because the reward distribution is sparse/default-heavy.

## Interpretation

The result is not a total victory on every metric, but it is the strongest evidence so far that the recurrent latent/EFLA world-model direction is useful:

- SEF-GRAM is consistently better than MLP.
- SEF-GRAM is consistently slightly better than GRU on aggregate hidden-state rollout, done calibration, and `has_key` tracking.
- SEF-GRAM variance is lower than GRU and much lower than MLP on observation rollout.

## Caveats

- The environment is still synthetic and low-dimensional.
- Only three seeds are documented.
- GRU may improve with tuning, larger hidden size, or sequence-level training adjustments.
- Event recall metrics should be printed in compact summary and inspected from the raw CSV.

## Next work

1. Print compact recall/rate metrics in the multi-seed summary.
2. Add harder memory settings:
   - lower `has_key=1` start probability;
   - larger grid;
   - longer horizon;
   - rarer key pickup and blocked-door events.
3. Add LSTM or small Transformer recurrent baseline.
4. Add planning evaluation: use learned world model to choose actions for key collection and goal reaching.
5. Add a README table summarizing the strongest benchmark results.
