import torch

from experiments.world_model_multistep_compare import (
    MultiStepCompareConfig,
    build_models,
    train_with_multistep_loss,
)


def test_build_multistep_compare_models():
    cfg = MultiStepCompareConfig(device="cpu", latent_dim=16, hidden_dim=32)
    models = build_models(cfg, torch.device("cpu"))
    labels = [label for label, _ in models]
    assert labels == ["sef_gram_multistep", "mlp_multistep"]


def test_train_with_multistep_loss_smoke():
    torch.manual_seed(0)
    cfg = MultiStepCompareConfig(
        steps=1,
        batch_size=4,
        eval_batches=1,
        rollout_horizon=2,
        device="cpu",
        latent_dim=16,
        hidden_dim=32,
    )
    label, model = build_models(cfg, torch.device("cpu"))[1]
    train_with_multistep_loss(model, cfg, label)
