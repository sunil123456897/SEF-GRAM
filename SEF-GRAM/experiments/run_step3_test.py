import torch
import math
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.scheduler import DirichletScheduler

def test_step3():
    print("=== Testing Step 3: Dirichlet Dynamic Lambda Phase Transition ===")
    
    cfg = WorldModelConfig(
        max_obs_dim=1,
        latent_dim=64,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=10,
        block_size_k=4
    )
    
    model = UniversalWorldModel(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    scheduler = DirichletScheduler(diversity_threshold=0.01, temperature=20.0, score_threshold=0.5, ema_decay=0.9)
    
    B, T = 4, 10
    
    print("\nStarting Training Loop...")
    print(f"{'Iter':>5} | {'Dirichlet':>10} | {'Lambda':>8} | {'ECHO Loss':>10} | {'Mean Reward':>12}")
    
    for i in range(150):
        # 1. Simulate Environment (Sine Wave)
        t_seq = torch.linspace(0, 2*math.pi, T).unsqueeze(0).repeat(B, 1)
        obs = torch.sin(t_seq).unsqueeze(-1)
        next_obs = torch.sin(t_seq + 0.1).unsqueeze(-1)
        
        # Simulate Dirichlet Energy drop over time (model learning "smoothness")
        noise_level = max(0.0, 1.0 - i/50.0) 
        obs = obs + torch.randn_like(obs) * noise_level
        next_obs = next_obs + torch.randn_like(next_obs) * noise_level
        
        # 2. Simulate RL Policy Behavior
        # As Lambda goes up, the RL agent "wakes up" and starts exploiting the environment.
        # Target action is 3 to maximize reward.
        lam = scheduler.lambda_gdpo
        if lam > 0.5:
            # Policy is optimizing!
            actions = torch.randint(2, 4, (B, T)) # Biased towards action 3
        else:
            # Policy is random
            actions = torch.randint(0, 4, (B, T))
            
        rewards = (actions == 3).float()
        advantages = rewards - rewards.mean()
        
        batch = WorldBatch(
            obs=obs,
            actions=actions,
            next_obs=next_obs,
            rewards=rewards,
            dones=torch.zeros(B, T),
            advantages=advantages
        )
        
        optimizer.zero_grad()
        
        # Pre-forward to get current Dirichlet energy
        with torch.no_grad():
            _, pre_metrics = model.loss(batch, lambda_gdpo=0.0)
            
        cur_dirichlet = pre_metrics["dirichlet_energy"].item()
        active_attractors = pre_metrics["active_attractors_count"].item() * cfg.env_vocab_size
        
        # Step the Safe Scheduler
        new_lam = scheduler.step(cur_dirichlet, active_attractors, cfg.env_vocab_size)
        
        # Real forward and backward with dynamic lambda
        loss, metrics = model.loss(batch, lambda_gdpo=new_lam)
        
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 10 == 0:
            print(f"{i+1:5d} | {cur_dirichlet:10.5f} | {new_lam:8.4f} | {metrics['echo_loss'].item():10.4f} | {rewards.mean().item():12.4f}")

if __name__ == "__main__":
    test_step3()
