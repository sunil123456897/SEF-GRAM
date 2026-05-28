from __future__ import annotations

import torch

from sef_gram.turn_based_env import MiniTerminalEnv
from sef_gram.turn_baselines import MLPTurnBaseline, GRUTurnBaseline


def test_mlp_turn_baseline_init_and_forward():
    env = MiniTerminalEnv(seed=0)
    model = MLPTurnBaseline(obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS, env_vocab_size=env.NUM_ENV_CLASSES, hidden_dim=32)
    batch = env.reset(4, torch.device("cpu"))
    z, memory, mu = model.initial_state(batch.obs)
    assert z.shape == (4, 32)
    logits = model.policy_logits(z)
    assert logits.shape == (4, env.NUM_ACTIONS)
    env_out = model.env_logits_from_z(z)
    assert env_out.shape == (4, env.NUM_ENV_CLASSES)
    next_batch = env.step(torch.zeros(4, dtype=torch.long))
    z2, mem2 = model.forward_turn(z, memory, torch.zeros(4, dtype=torch.long), obs=next_batch.obs)
    assert z2.shape == (4, 32)


def test_gru_turn_baseline_init_and_forward():
    env = MiniTerminalEnv(seed=0)
    model = GRUTurnBaseline(obs_dim=env.OBS_DIM, num_actions=env.NUM_ACTIONS, env_vocab_size=env.NUM_ENV_CLASSES, hidden_dim=32)
    batch = env.reset(4, torch.device("cpu"))
    z, memory, mu = model.initial_state(batch.obs)
    assert z.shape == (4, 32)
    logits = model.policy_logits(z)
    assert logits.shape == (4, env.NUM_ACTIONS)
    next_batch = env.step(torch.zeros(4, dtype=torch.long))
    z2, mem2 = model.forward_turn(z, memory, torch.zeros(4, dtype=torch.long), obs=next_batch.obs)
    assert z2.shape == (4, 32)


def test_compare_smoke():
    from experiments.echo_compare import CompareConfig, build_model, train_one_model, evaluate_model
    from sef_gram.echo_trainer import ECHOGDPOConfig

    device = torch.device("cpu")
    env = MiniTerminalEnv(seed=0)
    cfg = CompareConfig(seeds=(42,), steps=5, batch_size=4, num_candidates=2, max_turns=3, latent_dim=16, hidden_dim=32, device="cpu")
    train_cfg = ECHOGDPOConfig(
        steps=5, batch_size=4, num_candidates=2, latent_dim=16, hidden_dim=32,
        max_turns=3, device="cpu", seed=42,
    )

    for label in ["echo_gram", "gru", "mlp"]:
        model = build_model(label, env, cfg, device)
        model, tracking = train_one_model(model, lambda: MiniTerminalEnv(seed=99), train_cfg, device, label)
        eval_metrics = evaluate_model(model, MiniTerminalEnv(seed=199), train_cfg, device, num_episodes=10)
        assert "success_rate" in eval_metrics
