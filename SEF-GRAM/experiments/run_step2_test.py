import torch
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig
from sef_gram.planner import HybridLatentPlanner

def test_hybrid_planner():
    print("=== Testing Step 2: BJEPA Hybrid Gradient Planner ===")
    
    cfg = WorldModelConfig(
        max_obs_dim=16,
        latent_dim=64,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=10, # 10 tokens in dictionary
    )
    
    model = UniversalWorldModel(cfg)
    model.eval() # Freeze weights
    
    planner = HybridLatentPlanner(model)
    
    # 1. Setup Environment concepts (Attractor Space)
    vocab = model.core.env_head.weight.detach() # [10, 64]
    
    z_0 = vocab[0].unsqueeze(0)        # [START] token
    z_bomb = vocab[1]                  # [BOMB] token
    z_goal = vocab[2]                  # [END] token
    
    memory_0 = model.core.initial_memory(1, z_0.device)
    
    print("Executing Hybrid Plan (Random Shooting -> Gradient Refinement)...")
    
    result = planner.plan(
        z_0=z_0,
        memory_0=memory_0,
        z_goal=z_goal,
        z_bomb=z_bomb,
        H=5,
        num_samples=200,
        lr=0.05,
        max_steps=30
    )
    
    print("\n--- Metrics ---")
    print(f"prior_energy_initial (после Random Shooting): {result['prior_energy_initial']:.4f}")
    print(f"prior_energy_final (после Gradient Refinement): {result['prior_energy_final']:.4f}")
    print(f"optimization_steps: {result['optimization_steps']}")
    
    # Check max cosine to bomb in trajectory
    traj = result['trajectory'][0, 1:] # [H, latent_dim]
    max_bomb_sim = torch.max(torch.nn.functional.cosine_similarity(traj, z_bomb.unsqueeze(0), dim=-1))
    
    print("\n--- Analysis ---")
    print(f"Максимальная близость к бомбе (Косинус): {max_bomb_sim.item():.4f}")
    if max_bomb_sim.item() < 0.5:
        print("[УСПЕХ] Траектория безопасно обогнула бомбу (сходство < 0.5)!")
    else:
        print("[ПРОВАЛ] Траектория задела бомбу!")
        
    final_z = traj[-1]
    dist_to_goal = torch.nn.functional.mse_loss(final_z, z_goal).item()
    print(f"MSE до цели в конце: {dist_to_goal:.4f}")
    
    if dist_to_goal < result['prior_energy_initial']:
        print("[УСПЕХ] Градиентный спуск существенно уточнил траекторию элиты!")

if __name__ == "__main__":
    test_hybrid_planner()
