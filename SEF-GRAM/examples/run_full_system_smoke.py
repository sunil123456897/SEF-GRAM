from pathlib import Path
import sys

import torch


# Allow direct execution from either the repository root or the SEF-GRAM/
# project directory without requiring editable installation first.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
_REPO_ROOT = _THIS_FILE.parents[2] if len(_THIS_FILE.parents) > 2 else _PROJECT_ROOT
for path in (_PROJECT_ROOT, _REPO_ROOT, _REPO_ROOT / "SEF-GRAM"):
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)

from sef_gram.full_system import LatentPoEPlanner, SEFGRAMObjective, build_tiny_system


def main():
    torch.manual_seed(42)
    model = build_tiny_system(
        input_dim=8,
        latent_dim=16,
        hidden_dim=32,
        num_actions=4,
        env_vocab_size=16,
        recursion_depth=3,
        num_trajectories=4,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    context = torch.randn(8, 8)
    target = torch.randn(8, 8)
    actions = torch.randint(0, 4, (8, 3))

    for step in range(5):
        optimizer.zero_grad()
        loss, metrics = SEFGRAMObjective.vjepa_latent_loss(model, context, target, actions)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        print(
            f"step={step + 1} "
            f"loss={metrics['total'].item():.4f} "
            f"pred_nll={metrics['pred_nll'].item():.4f} "
            f"kl={metrics['kl'].item():.4f} "
            f"var={metrics['latent_variance'].item():.4f}"
        )

    planner = LatentPoEPlanner(model)
    result = planner.plan(context[:2])
    print("best action sequences:")
    print(result["best_actions"])


if __name__ == "__main__":
    main()
