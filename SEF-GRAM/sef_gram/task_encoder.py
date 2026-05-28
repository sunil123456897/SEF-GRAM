import torch
import torch.nn as nn

class TaskEncoder(nn.Module):
    """
    Amortized Task Encoder (Meta-Pretraining).
    Maps Support Set (obs, next_obs) -> Latent Slot Memory (Z_rule).
    """
    def __init__(self, latent_dim: int, num_slots: int = 4):
        super().__init__()
        self.num_slots = num_slots
        self.latent_dim = latent_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, latent_dim * num_slots)
        )
        
    def forward(self, support_obs: torch.Tensor, support_next_obs: torch.Tensor) -> torch.Tensor:
        """
        support_obs: [B, K, T, latent_dim] or [B, K*T, latent_dim]
        support_next_obs: [B, K, T, latent_dim] or [B, K*T, latent_dim]
        Returns: [B, num_slots, latent_dim]
        """
        B = support_obs.size(0)
        
        if support_obs.dim() == 4:
            support_obs = support_obs.view(B, -1, self.latent_dim)
            support_next_obs = support_next_obs.view(B, -1, self.latent_dim)
            
        x = torch.cat([support_obs, support_next_obs], dim=-1) # [B, N, 2*D]
        h = self.mlp(x) # [B, N, num_slots * D]
        
        h_pool = h.mean(dim=1) # [B, num_slots * D]
        slots = h_pool.view(B, self.num_slots, self.latent_dim)
        
        if self.training:
            if torch.rand(1).item() < 0.1:
                drop_idx = torch.randint(0, self.num_slots, (B,))
                for b in range(B):
                    slots[b, drop_idx[b]] = 0.0
                    
        return slots
