import torch

from experiments.world_model_eval import EvalConfig, evaluate_one_step, evaluate_open_loop_proxy, merge_rows, train_model
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig


def test_mlp_world_model_loss_and_predict():
    from sef_gram.world_envs import build_default_mixed_batcher

    torch.manual_seed(0)
    device = torch.device("cpu")
    batch = build_default_mixed_batcher(max_obs_dim=16).sample(4, device)
    model = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=16, hidden_dim=32, num_actions=4))
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    assert set(metrics) == {"total", "obs_mse", "reward_mse", "done_bce"}
    loss.backward()
    pred = model.predict_step(batch.obs, batch.actions)
    assert pred["next_obs"].shape == (4, 16)
    assert pred["reward"].shape == (4,)
    assert pred["done_prob"].shape == (4,)


def test_world_eval_metrics_smoke():
    torch.manual_seed(0)
    cfg = EvalConfig(steps=1, batch_size=4, eval_batches=2, rollout_horizon=2, device="cpu", latent_dim=16, hidden_dim=32)
    model = UniversalWorldModel(WorldModelConfig(max_obs_dim=16, latent_dim=16, hidden_dim=32, num_actions=4))
    train_model(model, cfg, "test_model")
    one = evaluate_one_step(model, cfg, "test_model")
    rollout = evaluate_open_loop_proxy(model, cfg, "test_model")
    row = merge_rows(one, rollout)
    assert row["model"] == "test_model"
    assert "one_step_total" in row
    assert "open_loop_obs_energy" in row
