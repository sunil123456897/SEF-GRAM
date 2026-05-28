import torch

from experiments.world_model_multistep_train import (
    MultiStepTrainConfig,
    differentiable_predict_step,
    multi_env_rollout_loss,
    rollout_loss_for_env,
)
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig
from sef_gram.world_rollout_envs import GridWorldRolloutEnv


def test_differentiable_predict_step_shapes():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    obs = torch.randn(4, 16)
    actions = torch.randint(0, 4, (4,))
    pred = differentiable_predict_step(model, obs, actions, max_obs_dim=16)
    assert pred["next_obs"].shape == (4, 16)
    assert pred["reward"].shape == (4,)
    assert pred["done_logit"].shape == (4,)


def test_rollout_loss_backward():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = MultiStepTrainConfig(batch_size=4, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    env = GridWorldRolloutEnv(max_obs_dim=16)
    loss, metrics = rollout_loss_for_env(model, env, cfg, device)
    assert torch.isfinite(loss)
    assert set(metrics) == {"rollout_obs_mse", "rollout_reward_mse", "rollout_done_bce"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads


def test_multi_env_rollout_loss_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = MultiStepTrainConfig(batch_size=4, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    loss, metrics = multi_env_rollout_loss(model, cfg, device)
    assert torch.isfinite(loss)
    assert set(metrics) == {"rollout_total", "rollout_obs_mse", "rollout_reward_mse", "rollout_done_bce"}
