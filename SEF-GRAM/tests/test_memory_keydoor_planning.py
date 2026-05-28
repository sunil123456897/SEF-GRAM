import torch

from experiments.memory_keydoor_planning_eval import (
    PlanningConfig,
    build_candidate_actions,
    execute_action_sequences,
    score_candidates_mlp,
)
from experiments.memory_keydoor_compare import MemoryKeyDoorSequenceEnv
from sef_gram.world_baselines import MLPWorldModel, MLPWorldModelConfig


def test_build_candidate_actions_shape():
    torch.manual_seed(0)
    device = torch.device("cpu")
    env = MemoryKeyDoorSequenceEnv(size=6, max_obs_dim=16)
    state = env.reset_state(4, device)
    obs = env.initial_obs(state)
    cfg = PlanningConfig(batch_size=4, plan_horizon=6, num_candidates=8, device="cpu", latent_dim=16, hidden_dim=32)
    candidates = build_candidate_actions(obs, cfg)
    assert candidates.shape == (4, 8, 6)
    assert candidates.dtype == torch.long
    assert int(candidates.min()) >= 0
    assert int(candidates.max()) < cfg.num_actions


def test_execute_action_sequences_metrics_shape():
    torch.manual_seed(0)
    device = torch.device("cpu")
    env = MemoryKeyDoorSequenceEnv(size=6, max_obs_dim=16)
    state = env.reset_state(4, device)
    actions = torch.randint(0, 4, (4, 5), device=device)
    result = execute_action_sequences(env, state, actions)
    assert result["total_reward"].shape == (4,)
    assert result["success"].shape == (4,)
    assert result["has_key"].shape == (4,)
    assert result["blocked_count"].shape == (4,)


def test_score_candidates_mlp_shape():
    torch.manual_seed(0)
    device = torch.device("cpu")
    env = MemoryKeyDoorSequenceEnv(size=6, max_obs_dim=16)
    state = env.reset_state(4, device)
    obs = env.initial_obs(state)
    cfg = PlanningConfig(batch_size=4, plan_horizon=5, num_candidates=6, device="cpu", latent_dim=16, hidden_dim=32)
    candidates = build_candidate_actions(obs, cfg)
    model = MLPWorldModel(MLPWorldModelConfig(max_obs_dim=16, hidden_dim=32, num_actions=4))
    scores = score_candidates_mlp(model, obs, candidates, cfg)
    assert scores.shape == (4, 6)
    assert torch.isfinite(scores).all()
