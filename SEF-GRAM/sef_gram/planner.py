import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class HybridLatentPlanner:
    """BJEPA Gradient Planner: Hybrid (Random Shooting + Gradient Refinement)
    
    Navigates the non-convex attractor-filled latent space using a warm-start
    followed by precise Adam optimization.
    """
    
    def __init__(self, model):
        self.model = model
        self.core = model.core
        self.action_dim = self.core.cfg.latent_dim

    def rollout_continuous(self, z_0: torch.Tensor, memory_0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """Rollout a continuous sequence of actions u_seq.
        
        Args:
            z_0: [B, latent_dim]
            memory_0: [B, latent_dim, latent_dim]
            u_seq: [B, H, action_dim]
            
        Returns:
            zs: [B, H+1, latent_dim] (includes initial state)
        """
        B, H, _ = u_seq.shape
        z = z_0
        memory = memory_0
        zs = [z]
        for t in range(H):
            action_vec = u_seq[:, t]
            # Bypass discrete Action Embedding and inject continuous vector directly
            z_action = self.core.transition.pre(torch.cat([z, action_vec], dim=-1))
            z, memory = self.core.transition.cell(z_action, memory)
            zs.append(z)
        return torch.stack(zs, dim=1)

    def plan(self, z_0: torch.Tensor, memory_0: torch.Tensor, z_goal: torch.Tensor, z_bomb: torch.Tensor, 
             H: int = 5, num_samples: int = 100, lr: float = 0.1, max_steps: int = 20) -> Dict:
        """Generate an optimal trajectory to reach z_goal while avoiding z_bomb."""
        
        device = z_0.device
        
        # --- PHASE 1: Random Shooting (Warm-start) ---
        # Duplicate states for batch rollout
        z_0_exp = z_0.repeat(num_samples, 1)
        memory_0_exp = memory_0.repeat(num_samples, 1, 1)
        
        # Sample random continuous actions
        u_samples = torch.randn(num_samples, H, self.action_dim, device=device)
        
        with torch.no_grad():
            zs_samples = self.rollout_continuous(z_0_exp, memory_0_exp, u_samples)
            
            # Energy = MSE(final, goal) + Penalty for hitting bomb
            z_H_samples = zs_samples[:, -1]
            mse_goal = F.mse_loss(z_H_samples, z_goal.unsqueeze(0).repeat(num_samples, 1), reduction='none').mean(dim=-1)
            
            # Bomb penalty (cosine similarity > margin)
            cosine_bomb = F.cosine_similarity(zs_samples[:, 1:], z_bomb.unsqueeze(0).unsqueeze(0), dim=-1)
            bomb_penalty = F.relu(cosine_bomb - 0.5).sum(dim=1)
            
            energy = mse_goal + 10.0 * bomb_penalty
            
        # Select Elite trajectory
        best_idx = torch.argmin(energy)
        u_elite = u_samples[best_idx:best_idx+1].clone()
        prior_energy_initial = energy[best_idx].item()
        
        # --- PHASE 2: Gradient Refinement ---
        u_seq = nn.Parameter(u_elite) # Initialize with elite trajectory
        optimizer = torch.optim.Adam([u_seq], lr=lr)
        
        for step in range(max_steps):
            optimizer.zero_grad()
            zs = self.rollout_continuous(z_0, memory_0, u_seq)
            
            mse_goal = F.mse_loss(zs[:, -1], z_goal.unsqueeze(0))
            cosine_bomb = F.cosine_similarity(zs[:, 1:], z_bomb.unsqueeze(0).unsqueeze(0), dim=-1)
            bomb_penalty = F.relu(cosine_bomb - 0.5).sum()
            
            loss = mse_goal + 10.0 * bomb_penalty
            loss.backward()
            optimizer.step()
            
        prior_energy_final = loss.item()
        
        return {
            "u_opt": u_seq.detach(),
            "trajectory": zs.detach(),
            "prior_energy_initial": prior_energy_initial,
            "prior_energy_final": prior_energy_final,
            "optimization_steps": max_steps
        }
