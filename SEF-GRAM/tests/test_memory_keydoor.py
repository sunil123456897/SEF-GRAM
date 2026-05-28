import torch

from experiments.memory_keydoor_compare import (
    MemoryKeyDoorConfig,
    MemoryKeyDoorSequenceEnv,
    evaluate_memory_keydoor,
    sequence_loss_mlp,
    sequence_loss_sef,
)
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig


def test_memory_keydoor_sequence_shapes():
    torch.manual_seed(0)
    device = torch.device("cpu")
    env = MemoryKeyDoorSequenceEnv(size=6, max_obs_dim=16)
    seq = env.sample_sequence(batch_size=4, horizon=3, device=device)
    assert seq["initial_obs"].shape == (4, 16)
    assert seq["obs"].shape == (4, 3, 16)
    assert seq["actions"].shape == (4, 3)
    assert seq["next_obs"].shape == (4, 3, 16)
    assert seq["rewards"].shape == (4, 3)
    assert seq["dones"].shape == (4, 3)
    assert torch.all(seq["obs"][:, 1:, 2:8] == 0.0)


def test_memory_keydoor_losses_backward():
    torch.manual_seed(0)
    cfg = MemoryKeyDoorConfig(batch_size=4, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    device = torch.device("cpu")
    env = MemoryKeyDoorSequenceEnv(size=6, max_obs_dim=16)
    seq = env.sample_sequence(cfg.batch_size, cfg.rollout_horizon, device)

    sef = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    sef_loss, sef_metrics = sequence_loss_sef(sef, seq, cfg)
    assert torch.isfinite(sef_loss)
    assert set(sef_metrics) == {"obs_mse", "reward_mse", "done_bce"}
    sef_loss.backward()
    assert [p.grad for p in sef.parameters() if p.grad is not None]

    mlp = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=16, hidden_dim=32, num_actions=4))
    mlp_loss, mlp_metrics = sequence_loss_mlp(mlp, seq, cfg)
    assert torch.isfinite(mlp_loss)
    assert set(mlp_metrics) == {"obs_mse", "reward_mse", "done_bce"}
    mlp_loss.backward()
    assert [p.grad for p in mlp.parameters() if p.grad is not None]


def test_memory_keydoor_eval_smoke():
    torch.manual_seed(0)
    cfg = MemoryKeyDoorConfig(batch_size=4, eval_batches=1, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    row = evaluate_memory_keydoor(model, "sef_gram_memory", cfg)
    assert row["model"] == "sef_gram_memory"
    assert row["env"] == "memory_key_door"
    assert "rollout_obs_mse_avg" in row
