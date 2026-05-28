# Memory Key-Door Planning Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

This document records the first model-based planning evaluation for the partially observable Memory Key-Door task.

The evaluation trains world models, generates candidate action sequences, scores them with each learned model, and executes the selected sequence in the real environment.

Compared policies/models:

- `random_policy`: random action sequence baseline.
- `oracle_candidate`: chooses the best candidate using the real environment. This is not a deployable policy; it measures whether the candidate set contains good plans.
- `sef_gram_memory`: recurrent latent/EFLA world model used for candidate scoring.
- `gru_memory`: recurrent GRU world model baseline used for candidate scoring.
- `mlp_memory`: memoryless MLP world model baseline used for candidate scoring.

## Initial sparse-scoring run

Earlier sparse model scoring used predicted reward, done probability, and key probability only. That failed to use the learned transition geometry effectively.

GPU run with sparse scoring:

| model | success | key_rate | avg_reward | avg_blocked |
|---|---:|---:|---:|---:|
| random_policy | 5.23% | 27.46% | -0.2719 | 0.4352 |
| oracle_candidate | 94.49% | 97.19% | 3.1937 | 0.0465 |
| sef_gram_memory | 4.65% | 24.92% | -0.3017 | 0.5098 |
| gru_memory | 2.70% | 18.98% | -0.3256 | 0.4859 |
| mlp_memory | 4.61% | 25.43% | -0.2946 | 0.4988 |

Interpretation:

- The high oracle result showed that the candidate set contained good plans.
- The learned models failed to rank those plans correctly using sparse reward/done/key scoring alone.
- This indicated a planning objective/scoring problem rather than a candidate-generation problem.

## Geometry-aware scoring run

The planner was updated to include dense geometry-aware progress terms:

```text
score += predicted_reward
       + progress toward key before predicted has_key
       + progress toward goal after predicted has_key
       + done_bonus
       + key_bonus
```

Command:

```bash
python experiments/memory_keydoor_planning_eval.py --steps 500 --batch-size 128 --eval-batches 20 --plan-horizon 20 --num-candidates 256 --export-csv results_memory_keydoor_planning.csv
```

Results:

| model | success | key_rate | avg_reward | avg_blocked |
|---|---:|---:|---:|---:|
| random_policy | 5.23% | 27.46% | -0.2719 | 0.4352 |
| oracle_candidate | 94.49% | 97.19% | 3.1937 | 0.0465 |
| sef_gram_memory | 23.32% | 86.88% | 0.3971 | 0.3148 |
| gru_memory | 16.88% | 85.98% | 0.2077 | 0.3473 |
| mlp_memory | 12.62% | 28.32% | -0.0078 | 0.4125 |

## CPU smoke run

Command:

```bash
python experiments/memory_keydoor_planning_eval.py --steps 20 --batch-size 16 --eval-batches 3 --plan-horizon 8 --num-candidates 32 --device cpu --latent-dim 32 --hidden-dim 64
```

Results with geometry-aware scoring:

| model | success | key_rate | avg_reward | avg_blocked |
|---|---:|---:|---:|---:|
| random_policy | 0.00% | 14.58% | -0.1446 | 0.2083 |
| oracle_candidate | 45.83% | 87.50% | 0.6804 | 0.0208 |
| sef_gram_memory | 2.08% | 10.42% | -0.1308 | 0.1875 |
| gru_memory | 0.00% | 29.17% | -0.1125 | 0.2083 |
| mlp_memory | 6.25% | 14.58% | -0.0746 | 0.3958 |

Interpretation:

- The CPU smoke run is too short and undertrained for strong conclusions.
- It remains useful for checking that the planning script runs end-to-end.

## Current conclusion

The geometry-aware planning result is a positive Phase 2 result:

1. SEF-GRAM clearly beats random policy:
   - success: 23.32% vs 5.23%.
   - average reward: 0.3971 vs -0.2719.
   - key rate: 86.88% vs 27.46%.
2. SEF-GRAM beats the recurrent GRU planner:
   - success: 23.32% vs 16.88%.
   - average reward: 0.3971 vs 0.2077.
   - fewer blocked moves: 0.3148 vs 0.3473.
3. SEF-GRAM beats the MLP planner strongly:
   - success: 23.32% vs 12.62%.
   - key rate: 86.88% vs 28.32%.
   - average reward: 0.3971 vs -0.0078.
4. The large gap to `oracle_candidate` means the planning layer is still far from optimal:
   - SEF-GRAM success: 23.32%.
   - oracle candidate success: 94.49%.

## Interpretation

This is not final proof of a universal architecture, but it is a strong Phase 2 milestone:

- The world model is useful for action selection, not only prediction.
- SEF-GRAM produces better planning outcomes than GRU and MLP under the same candidate set/scoring design.
- Planning remains the weakest component and should be improved separately.

## Caveats

- Only one GPU seed is documented for planning.
- The planner uses hand-designed geometry-aware scoring, not a learned value function.
- Candidate generation includes visible-fact path priors.
- The environment is still synthetic and low-dimensional.
- The gap to oracle indicates substantial headroom.

## Next work

1. Add multi-seed planning evaluation.
2. Add a learned value/planning head trained from rollouts.
3. Add CEM/MPC candidate optimization instead of mostly random shooting.
4. Add a no-op action or explicit stop action to reduce random padding artifacts.
5. Add harder planning settings after the current planner is stable.
