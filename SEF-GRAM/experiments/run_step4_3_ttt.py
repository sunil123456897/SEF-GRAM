import torch
import torch.nn as nn
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder, generate_shift_trajectory, generate_invert_trajectory

def run_ttt():
    print("=== Phase 4.3: Test-Time Training (BJEPA MPC on Meta-Dynamics) ===")
    
    latent_dim = 64
    cfg = WorldModelConfig(
        max_obs_dim=latent_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=10, 
        block_size_k=4
    )
    
    model = UniversalWorldModel(cfg)
    encoder = ARCGridEncoder(latent_dim)
    decoder = ARCGridDecoder(latent_dim)
    
    # --- Meta-Dynamics Pretraining (Phase 4.2b) ---
    print("\n[Phase 4.2b] Pretraining Meta-Dynamics with random Z_rule vectors...")
    params = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    B, T_full = 2, 10
    
    for i in range(250):
        # We mix both tasks in the batch
        grids_shift = generate_shift_trajectory(B // 2, T_full+1)
        grids_invert = generate_invert_trajectory(B // 2, T_full+1)
        grids = torch.cat([grids_shift, grids_invert], dim=0)
        
        # Consistent Z_rule for each task
        torch.manual_seed(42)
        z_shift = torch.randn(1, latent_dim).expand(B//2, -1)
        torch.manual_seed(99)
        z_invert = torch.randn(1, latent_dim).expand(B//2, -1)
        torch.seed()
        task_emb_gt = torch.cat([z_shift, z_invert], dim=0)
        
        obs_grids = grids[:, :-1]
        next_obs_grids = grids[:, 1:]
        
        obs = encoder(obs_grids)
        next_obs = encoder(next_obs_grids)
        
        batch = WorldBatch(
            obs=obs,
            actions=torch.zeros(B, T_full, dtype=torch.long),
            next_obs=next_obs,
            rewards=torch.zeros(B, T_full),
            dones=torch.zeros(B, T_full),
            task_emb=task_emb_gt
        )
        
        optimizer.zero_grad()
        loss, _ = model.loss(batch, lambda_gdpo=0.0)
        
        out = model.forward(batch)
        grid_logits = decoder(out["z_preds"])
        t_start = out["t_start"]
        K = out["K"]
        target_grids = next_obs_grids[:, t_start:t_start+K]
        ce_loss = F.cross_entropy(grid_logits.reshape(-1, 10, 30, 30), target_grids.reshape(-1, 30, 30))
        
        total_loss = loss + ce_loss
        total_loss.backward()
        optimizer.step()

    print("[Phase 4.2b] Pretraining Complete.")
    
    # --- TTT Inference (Phase 4.3) ---
    print("\n[Phase 4.3] Starting Test-Time Training (TTT) Inference...")
    
    for p in model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    model.eval()
    encoder.eval()
    decoder.eval()
    
    # Support Set: 3 examples of Shift
    support_grids = generate_shift_trajectory(3, T_full+1)
    # Query Set: 1 example of Shift
    query_grids = generate_shift_trajectory(1, T_full+1)
    
    # 1. Initialize Z_rule
    Z_rule = nn.Parameter(torch.randn(1, latent_dim))
    optimizer_ttt = torch.optim.Adam([Z_rule], lr=0.05)
    
    print("\nOptimizing Z_rule on Support Set...")
    for step in range(50):
        optimizer_ttt.zero_grad()
        
        s_obs_grids = support_grids[:, :-1]
        s_next_obs_grids = support_grids[:, 1:]
        
        obs_enc = encoder(s_obs_grids)
        target_enc = encoder(s_next_obs_grids)
        
        z = model.core.encode_context(obs_enc[:, 0], sample=False)["z"]
        memory = model.core.initial_memory(3, z.device)
        
        Z_rule_exp = Z_rule.expand(3, -1)
        
        loss_ttt = 0
        for t in range(T_full):
            step_dict = model.core.transition(z, torch.zeros(3, dtype=torch.long), memory, task_emb=Z_rule_exp)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            loss_ttt += F.mse_loss(z, target_enc[:, t].detach())
            
        loss_ttt.backward()
        optimizer_ttt.step()
        
        if (step+1) % 10 == 0:
            print(f"TTT Step {step+1:2d} | Support Energy Loss (BJEPA): {loss_ttt.item():.4f}")
            
    print("\n[VERIFICATION] Applying optimized Z_rule to Query Set...")
    with torch.no_grad():
        q_obs_grids = query_grids[:, :-1]
        q_next_obs_grids = query_grids[:, 1:]
        
        q_obs_enc = encoder(q_obs_grids)
        z = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
        memory = model.core.initial_memory(1, z.device)
        
        Z_rule_exp = Z_rule.expand(1, -1)
        
        zs = []
        for t in range(T_full):
            step_dict = model.core.transition(z, torch.zeros(1, dtype=torch.long), memory, task_emb=Z_rule_exp)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            zs.append(z)
            
        zs = torch.stack(zs, dim=1)
        grid_logits = decoder(zs)
        preds = grid_logits.argmax(dim=2)
        
        acc = (preds == q_next_obs_grids).float().mean()
        print(f"Query Final Accuracy: {acc.item()*100:.1f}%")
        
        if acc.item() > 0.8:
            print("[SUCCESS] Z_rule captured the Meta-Physics and successfully solved the Query!")
        else:
            print("[WARNING] Query accuracy is low.")

if __name__ == "__main__":
    run_ttt()
