from __future__ import annotations

import torch

from sef_gram.turn_based_env import MiniTerminalEnv
from sef_gram.echo_trainer import (
    ECHOGDPOConfig,
    ECHOGRAMWrapper,
    _gdpo_advantages_from_rewards,
    run_echo_gdpo_step,
    train_echo_gdpo,
)


def test_mini_terminal_env_shapes():
    env = MiniTerminalEnv(seed=0)
    B = 8
    batch = env.reset(B, torch.device("cpu"))
    assert batch.obs.shape == (B, env.OBS_DIM)
    assert batch.env_target.shape == (B,)
    assert batch.reward.shape == (B,)
    assert batch.done.shape == (B,)
    assert batch.valid_actions.shape == (B, env.NUM_ACTIONS)
    assert batch.valid_actions.dtype == torch.bool


def test_mini_terminal_env_ls_reveals_files():
    env = MiniTerminalEnv(seed=0)
    B = 4
    _ = env.reset(B, torch.device("cpu"))
    ls_action = torch.zeros(B, dtype=torch.long)
    batch = env.step(ls_action)
    assert env._visible.all()
    assert not batch.done.any()


def test_mini_terminal_env_read_reveals_content():
    env = MiniTerminalEnv(seed=0)
    B = 4
    _ = env.reset(B, torch.device("cpu"))
    env.step(torch.zeros(B, dtype=torch.long))
    read_action = torch.ones(B, dtype=torch.long)
    batch = env.step(read_action)
    assert (batch.env_target >= 1).all()
    assert (batch.env_target <= env.NUM_ENV_CLASSES - 1).all()


def test_mini_terminal_env_answer_gives_reward():
    env = MiniTerminalEnv(seed=0)
    B = 1
    _ = env.reset(B, torch.device("cpu"))
    env.step(torch.zeros(1, dtype=torch.long))
    contents = env._contents[0]
    target = env._target[0].item()
    correct_file = (contents == target).nonzero(as_tuple=True)[0][0].item()
    answer_action = torch.tensor([env.num_files + 1 + correct_file], dtype=torch.long)
    batch = env.step(answer_action)
    assert batch.reward[0].item() == 1.0
    assert batch.done[0].item()


def test_gdpo_advantages_from_rewards():
    rewards = torch.tensor([[0.0, 1.0, 0.0, 5.0], [2.0, 2.0, 2.0, 2.0]])
    adv = _gdpo_advantages_from_rewards(rewards)
    assert adv.shape == rewards.shape
    assert adv[1].abs().max().item() < 1e-6


def test_echogram_wrapper_creates():
    model = ECHOGRAMWrapper(
        obs_dim=15, num_actions=11, env_vocab_size=6, latent_dim=16, hidden_dim=32
    )
    assert model.core.env_head is not None
    obs = torch.randn(4, 15)
    z, memory, mu = model.initial_state(obs)
    assert z.shape == (4, 16)
    assert memory.shape == (4, 16, 16)


def test_run_echo_gdpo_step_smoke():
    torch.manual_seed(0)
    cfg = ECHOGDPOConfig(
        steps=1, batch_size=4, num_candidates=2,
        latent_dim=16, hidden_dim=32, max_turns=3,
        device="cpu", seed=0,
    )
    env = MiniTerminalEnv(seed=0)
    model = ECHOGRAMWrapper(
        obs_dim=env.OBS_DIM,
        num_actions=env.NUM_ACTIONS,
        env_vocab_size=env.NUM_ENV_CLASSES,
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    )
    loss, metrics = run_echo_gdpo_step(model, env, cfg, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert "policy_loss" in metrics
    assert "env_loss" in metrics
    assert "success_rate" in metrics
    assert "dirichlet_energy" in metrics


def test_echo_gdpo_train_smoke():
    torch.manual_seed(0)
    cfg = ECHOGDPOConfig(
        steps=3, batch_size=4, num_candidates=2,
        latent_dim=16, hidden_dim=32, max_turns=3,
        device="cpu", seed=0,
    )
    model = train_echo_gdpo(cfg)
    assert model is not None
