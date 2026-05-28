import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridTTTPlanner:
    """
    Hybrid Test-Time Training Planner (Deep BJEPA MPC).
    Random Restart -> Top-K Filter -> Deep Adam Optimization -> Majority Vote.
    """
    def __init__(self, world_model, latent_dim: int, num_slots: int = 4, num_hypotheses: int = 64, top_k: int = 4):
        self.model = world_model
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.num_hypotheses = num_hypotheses
        self.top_k = top_k
        
    def plan(self, support_obs, support_target, T: int, z_amortized: torch.Tensor, lr: float = 0.05, max_steps: int = 50, patience: int = 5):
        K_supp = support_obs.size(0)
        device = support_obs.device
        
        # 1. Initialize N hypotheses with Amortized Warm-Start
        Z_init = torch.randn(self.num_hypotheses, self.num_slots, self.latent_dim, device=device)
        Z_init[0] = z_amortized[0]
        if self.num_hypotheses >= 16:
            Z_init[1:16] = z_amortized[0].unsqueeze(0) + torch.randn(15, self.num_slots, self.latent_dim, device=device) * 0.1
            
        Z_rules = nn.Parameter(Z_init)
        opt = torch.optim.Adam([Z_rules], lr=lr, weight_decay=1e-4)
        
        # 2. Run 10 steps of Adam on all 64
        for step in range(10):
            opt.zero_grad()
            loss = self._compute_loss(Z_rules, support_obs, support_target, T)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z_rules], 1.0)
            opt.step()
            
        # 3. Filter top-K
        with torch.no_grad():
            losses = self._compute_per_hypothesis_loss(Z_rules, support_obs, support_target, T)
            top_indices = torch.topk(losses, self.top_k, largest=False).indices
            Z_top = Z_rules[top_indices].detach().clone()
            
        # 4. Continue optimization for top-K
        Z_top.requires_grad_(True)
        opt_top = torch.optim.Adam([Z_top], lr=lr, weight_decay=1e-4)
        
        best_Z = Z_top.clone()
        best_loss = float('inf')
        current_patience = 0
        
        for step in range(max_steps - 10):
            opt_top.zero_grad()
            loss = self._compute_loss(Z_top, support_obs, support_target, T)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Z_top], 1.0)
            opt_top.step()
            
            val = loss.item()
            if val < best_loss - 1e-4:
                best_loss = val
                best_Z = Z_top.detach().clone()
                current_patience = 0
            else:
                current_patience += 1
                
            if current_patience >= patience or val < 0.1:
                break
                
        return best_Z # [TopK, Slots, D]
        
    def _compute_loss(self, Z, obs, target, T):
        N = Z.size(0)
        K = obs.size(0)
        
        obs_exp = obs.unsqueeze(0).expand(N, K, -1).reshape(N*K, -1)
        z = self.model.core.encode_context(obs_exp, sample=False)["z"]
        memory = self.model.core.initial_memory(N*K, z.device)
        
        Z_exp = Z.unsqueeze(1).expand(N, K, self.num_slots, self.latent_dim).reshape(N*K, self.num_slots, self.latent_dim)
        target_exp = target.unsqueeze(0).expand(N, K, T, -1).reshape(N*K, T, -1)
        
        loss = 0
        for t in range(T):
            step_dict = self.model.core.transition(z, torch.zeros(N*K, dtype=torch.long, device=z.device), memory, task_emb=Z_exp)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            loss += F.mse_loss(z, target_exp[:, t].detach())
            
        return loss
        
    def _compute_per_hypothesis_loss(self, Z, obs, target, T):
        N = Z.size(0)
        K = obs.size(0)
        
        obs_exp = obs.unsqueeze(0).expand(N, K, -1).reshape(N*K, -1)
        z = self.model.core.encode_context(obs_exp, sample=False)["z"]
        memory = self.model.core.initial_memory(N*K, z.device)
        Z_exp = Z.unsqueeze(1).expand(N, K, self.num_slots, self.latent_dim).reshape(N*K, self.num_slots, self.latent_dim)
        target_exp = target.unsqueeze(0).expand(N, K, T, -1).reshape(N*K, T, -1)
        
        losses = torch.zeros(N, device=Z.device)
        
        for t in range(T):
            step_dict = self.model.core.transition(z, torch.zeros(N*K, dtype=torch.long, device=z.device), memory, task_emb=Z_exp)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            
            mse = F.mse_loss(z, target_exp[:, t], reduction='none').mean(dim=-1) # [N*K]
            mse = mse.view(N, K).mean(dim=1) # [N]
            losses += mse
            
        return losses
