# Key-Door Multi-Step Comparison Results

Owner: solo research project
Last updated: 2026-05-28
Expires/recheck by: 2026-06-11

## Status

The mixed world-model benchmark now includes:

- GridWorld
- KeyDoorGridWorld
- LinePhysics

The fair multi-step comparison trains both SEF-GRAM and the MLP baseline with the same objective:

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

| model | env | h1_obs_mse | h3_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.3158 | 0.3102 | 0.3085 | 0.1912 |
| sef_gram_multistep | key_door_gridworld | 0.3389 | 0.3211 | 0.3250 | 0.1567 |
| sef_gram_multistep | line_physics | 0.2628 | 0.2893 | 0.2800 | 0.4209 |
| mlp_multistep | gridworld | 0.3865 | 0.3736 | 0.3800 | 0.0759 |
| mlp_multistep | key_door_gridworld | 0.3484 | 0.3518 | 0.3504 | 0.0409 |
| mlp_multistep | line_physics | 0.2462 | 0.2527 | 0.2500 | 0.6747 |

Interpretation:

- The CPU smoke run is too short for final conclusions.
- SEF-GRAM has better observation rollout error on GridWorld and KeyDoorGridWorld.
- MLP has better reward prediction on GridWorld and KeyDoorGridWorld.
- SEF-GRAM has better LinePhysics reward prediction, while MLP has better short-horizon observation error in this small run.

## GPU run

Command:

```bash
python experiments/world_model_multistep_compare.py --steps 500 --batch-size 128 --eval-batches 20 --rollout-horizon 10
```

Results:

| model | env | h1_obs_mse | h10_obs_mse | avg_obs_mse | avg_reward_mse |
|---|---|---:|---:|---:|---:|
| sef_gram_multistep | gridworld | 0.0405 | 0.0876 | 0.0691 | 0.0523 |
| sef_gram_multistep | key_door_gridworld | 0.0856 | 0.1223 | 0.1094 | 0.0197 |
| sef_gram_multistep | line_physics | 0.0179 | 0.0709 | 0.0485 | 0.1200 |
| mlp_multistep | gridworld | 0.0319 | 0.0775 | 0.0632 | 0.0650 |
| mlp_multistep | key_door_gridworld | 0.0676 | 0.1055 | 0.0945 | 0.0282 |
| mlp_multistep | line_physics | 0.0185 | 0.1553 | 0.0757 | 0.2444 |

## Current conclusion

The result is mixed and informative:

1. MLP is stronger on observation prediction for the discrete GridWorld-style environments, including KeyDoorGridWorld.
2. SEF-GRAM is stronger on reward prediction across all three environments in the GPU run.
3. SEF-GRAM remains clearly stronger on continuous long-horizon dynamics in LinePhysics.
4. KeyDoorGridWorld is still not hard enough to force a clear SEF-GRAM advantage in observation rollout; the full state is visible, so MLP can learn the transition rule directly.

## Implication

The next environment should hide part of the state or require temporal memory. A fully observed key-door environment is still close to a deterministic transition-regression problem. To test the world-model hypothesis more directly, the next tasks should include partial observability or delayed reward:

- HiddenTargetGridWorld: target/key is shown only at reset or through a clue token.
- KeyDoorMemoryGridWorld: key/door/goal facts are observed earlier, then removed from the current observation.
- DelayedRewardLinePhysics: reward depends on target shown earlier but hidden later.

## Next work

1. Add partial-observation/memory variants of KeyDoorGridWorld.
2. Track `has_key` accuracy separately instead of only aggregate observation MSE.
3. Add per-dimension error metrics for causal state variables.
4. Add multi-seed mean/std for fair comparison.
5. Add planning evaluation using the learned world model.
