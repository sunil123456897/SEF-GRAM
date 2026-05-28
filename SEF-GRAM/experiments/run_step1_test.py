import torch
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch

def test_integration():
    print("=== Testing Diffusion-GRAM & EFLA Attractors Integration ===")
    
    cfg = WorldModelConfig(
        max_obs_dim=16,
        latent_dim=64,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=100,
        block_size_k=4,
        ema_decay=0.99
    )
    
    model = UniversalWorldModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy sequence batch [B=2, T=10]
    B, T = 2, 10
    obs = torch.randn(B, T, 16)
    actions = torch.randint(0, 4, (B, T))
    next_obs = torch.randn(B, T, 16)
    rewards = torch.rand(B, T)
    dones = torch.zeros(B, T)
    
    batch = WorldBatch(obs=obs, actions=actions, next_obs=next_obs, rewards=rewards, dones=dones)
    
    print("\nStarting 100 iterations of training to check metrics...")
    
    for i in range(100):
        optimizer.zero_grad()
        loss, metrics = model.loss(batch)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 20 == 0:
            print(f"Iter {i+1:3d} | Total Loss: {metrics['total']:.4f} | "
                  f"Continuity Cosine: {metrics['latent_continuity_cosine']:.4f} | "
                  f"Active Attractors: {metrics['active_attractors_count']:.1%}")

if __name__ == "__main__":
    test_integration()
