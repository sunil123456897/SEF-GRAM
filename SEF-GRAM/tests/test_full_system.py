import torch

from sef_gram.full_system import (
    LatentPoEPlanner,
    SEFGRAMObjective,
    build_tiny_system,
)


def test_vjepa_latent_loss_backward():
    torch.manual_seed(0)
    model = build_tiny_system(input_dim=8, latent_dim=16, hidden_dim=32, num_actions=4, recursion_depth=3)
    context = torch.randn(5, 8)
    target = torch.randn(5, 8)
    actions = torch.randint(0, 4, (5, 3))

    loss, metrics = SEFGRAMObjective.vjepa_latent_loss(model, context, target, actions)

    assert torch.isfinite(loss)
    assert metrics["latent_variance"].item() >= 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "expected gradients to flow through the integrated system"


def test_recursive_rollout_shapes_and_planner():
    torch.manual_seed(1)
    model = build_tiny_system(input_dim=8, latent_dim=16, hidden_dim=32, num_actions=5, recursion_depth=4, num_trajectories=3)
    context = torch.randn(2, 8)

    rollout = model.recursive_rollout(context)
    assert rollout["latents"].shape == (2, 3, 4, 16)
    assert rollout["actions"].shape == (2, 3, 4)
    assert rollout["policy_logits"].shape == (2, 3, 4, 5)

    planner = LatentPoEPlanner(model)
    result = planner.plan(context)
    assert result["best_actions"].shape == (2, 4)
    assert result["scores"].shape == (2, 3)


def test_gdpo_and_echo_loss():
    torch.manual_seed(2)
    advantages = SEFGRAMObjective.gdpo_advantages(
        {
            "correctness": torch.tensor([0.0, 1.0, 0.0, 1.0]),
            "format": torch.tensor([0.0, 0.5, 1.0, 1.0]),
            "brevity": torch.tensor([0.8, 0.8, 0.6, 0.4]),
        },
        weights={"correctness": 2.0, "format": 0.5, "brevity": 0.2},
    )
    assert advantages.shape == (4,)
    assert abs(advantages.mean().item()) < 1e-5

    policy_logits = torch.randn(4, 6, 5, requires_grad=True)
    actions = torch.randint(0, 5, (4, 6))
    env_logits = torch.randn(4, 6, 11, requires_grad=True)
    env_targets = torch.randint(0, 11, (4, 6))

    loss, metrics = SEFGRAMObjective.echo_gdpo_loss(
        policy_logits,
        actions,
        advantages,
        env_logits=env_logits,
        env_targets=env_targets,
        echo_lambda=0.1,
        entropy_coef=0.01,
    )
    assert torch.isfinite(loss)
    assert metrics["env_loss"].item() > 0.0
    loss.backward()
    assert policy_logits.grad is not None
    assert env_logits.grad is not None
