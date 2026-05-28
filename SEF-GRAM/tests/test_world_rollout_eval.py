import torch

from experiments.world_model_rollout_eval import RolloutEvalConfig, evaluate_true_rollout, train_model
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig
from sef_gram.world_rollout_envs import GridWorldRolloutEnv, KeyDoorGridWorldRolloutEnv, LinePhysicsRolloutEnv, rollout_envs


def test_rollout_envs_step_shapes():
    torch.manual_seed(0)
    device = torch.device("cpu")
    for env in (
        GridWorldRolloutEnv(max_obs_dim=16),
        KeyDoorGridWorldRolloutEnv(max_obs_dim=16),
        LinePhysicsRolloutEnv(max_obs_dim=16),
    ):
        obs = env.reset(4, device)
        actions = env.sample_actions(4, device)
        batch = env.step_from_obs(obs, actions)
        assert batch.obs.shape == (4, 16)
        assert batch.next_obs.shape == (4, 16)
        assert batch.actions.shape == (4,)
        assert batch.rewards.shape == (4,)
        assert batch.dones.shape == (4,)
    assert set(rollout_envs(max_obs_dim=16)) == {"gridworld", "key_door_gridworld", "line_physics"}


def test_true_rollout_eval_smoke():
    torch.manual_seed(0)
    cfg = RolloutEvalConfig(steps=1, batch_size=4, eval_batches=2, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    train_model(model, cfg, "test_sef")
    rows = evaluate_true_rollout(model, cfg, "test_sef")
    assert len(rows) == 3
    for row in rows:
        assert row["model"] == "test_sef"
        assert row["env"] in {"gridworld", "key_door_gridworld", "line_physics"}
        assert "rollout_obs_mse_avg" in row


def test_mlp_true_rollout_eval_smoke():
    torch.manual_seed(0)
    cfg = RolloutEvalConfig(steps=1, batch_size=4, eval_batches=1, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=16, hidden_dim=32, num_actions=4))
    rows = evaluate_true_rollout(model, cfg, "mlp")
    assert len(rows) == 3
