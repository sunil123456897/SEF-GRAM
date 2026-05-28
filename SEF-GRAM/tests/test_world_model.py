import torch

from sef_gram.world_envs import GridWorldBatcher, GridWorldSpec, LinePhysicsBatcher, LinePhysicsSpec, build_default_mixed_batcher
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, pad_obs


def test_pad_obs():
    x = torch.ones(2, 3)
    y = pad_obs(x, 5)
    assert y.shape == (2, 5)
    assert torch.allclose(y[:, :3], x)
    assert torch.allclose(y[:, 3:], torch.zeros(2, 2))


def test_world_env_batchers():
    device = torch.device("cpu")
    grid = GridWorldBatcher(GridWorldSpec(size=5, max_obs_dim=16)).sample(8, device)
    physics = LinePhysicsBatcher(LinePhysicsSpec(max_obs_dim=16)).sample(8, device)
    mixed = build_default_mixed_batcher(max_obs_dim=16).sample(8, device)
    for batch in (grid, physics, mixed):
        assert batch.obs.shape == (8, 16)
        assert batch.next_obs.shape == (8, 16)
        assert batch.actions.shape == (8,)
        assert batch.rewards.shape == (8,)
        assert batch.dones.shape == (8,)


def test_universal_world_model_loss_backward():
    torch.manual_seed(0)
    device = torch.device("cpu")
    batch = build_default_mixed_batcher(max_obs_dim=16).sample(8, device)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert set(metrics) == {"total", "latent_mse", "latent_logvar_reg", "kl", "obs_mse", "reward_mse", "done_bce"}
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads


def test_universal_world_model_predict_step():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    obs = torch.randn(4, 16)
    actions = torch.randint(0, 4, (4,))
    pred = model.predict_step(obs, actions)
    assert pred["next_obs"].shape == (4, 16)
    assert pred["reward"].shape == (4,)
    assert pred["done_prob"].shape == (4,)
