import torch
import torch.nn as nn
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder, generate_composite_trajectory
from sef_gram.task_encoder import TaskEncoder
from sef_gram.ttt_planner import HybridTTTPlanner

def run_scale_ttt():
    print("=== Phase 5: Production Scaling (Latent Slot Memory & Hybrid TTT) ===")
    
    latent_dim = 64
    num_slots = 4
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
    task_encoder = TaskEncoder(latent_dim, num_slots=num_slots)
    
    print("\n[Phase 5.1] Pretraining Meta-Dynamics with Amortized Task Encoder...")
    params = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(task_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    B, T_full = 4, 10
    
    for i in range(250):
        grids = generate_composite_trajectory(B, T_full+1)
        
        obs_grids = grids[:, :-1]
        next_obs_grids = grids[:, 1:]
        
        obs = encoder(obs_grids)
        next_obs = encoder(next_obs_grids)
        
        z_rule = task_encoder(obs, next_obs)
        
        batch = WorldBatch(
            obs=obs,
            actions=torch.zeros(B, T_full, dtype=torch.long),
            next_obs=next_obs,
            rewards=torch.zeros(B, T_full),
            dones=torch.zeros(B, T_full),
            task_emb=z_rule
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
        
        if (i+1) % 50 == 0:
            print(f"Iter {i+1:3d} | VJEPA: {loss.item():.4f} | CE: {ce_loss.item():.4f}")

    print("[Phase 5.1] Pretraining Complete.")
    
    print("\n[Phase 5.2] Starting Inference Comparisons...")
    
    for p in model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    for p in task_encoder.parameters(): p.requires_grad = False
    
    model.eval()
    encoder.eval()
    decoder.eval()
    task_encoder.eval()
    
    support_grids = generate_composite_trajectory(3, T_full+1)
    query_grids = generate_composite_trajectory(1, T_full+1)
    
    s_obs_grids = support_grids[:, :-1]
    s_next_obs_grids = support_grids[:, 1:]
    s_obs_enc = encoder(s_obs_grids)
    s_target_enc = encoder(s_next_obs_grids)
    
    q_obs_grids = query_grids[:, :-1]
    q_next_obs_grids = query_grids[:, 1:]
    q_obs_enc = encoder(q_obs_grids)
    
    print("\n--- Baseline: Amortized Task Encoder ---")
    with torch.no_grad():
        s_obs_enc_b = s_obs_enc.unsqueeze(0)
        s_target_enc_b = s_target_enc.unsqueeze(0)
        
        z_rule_amortized = task_encoder(s_obs_enc_b, s_target_enc_b)
        
        z = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
        memory = model.core.initial_memory(1, z.device)
        
        zs = []
        for t in range(T_full):
            step_dict = model.core.transition(z, torch.zeros(1, dtype=torch.long), memory, task_emb=z_rule_amortized)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            zs.append(z)
            
        zs = torch.stack(zs, dim=1)
        grid_logits = decoder(zs)
        preds = grid_logits.argmax(dim=2)
        
        acc_amortized = (preds == q_next_obs_grids).float().mean()
        print(f"Amortized Task Encoder Accuracy: {acc_amortized.item()*100:.1f}%")
        
    print("\n--- Advanced: Hybrid TTT Planner ---")
    planner = HybridTTTPlanner(model, latent_dim, num_slots=num_slots, num_hypotheses=64, top_k=4)
    
    z_rule_ttt = planner.plan(s_obs_enc[:, 0], s_target_enc, T=T_full, z_amortized=z_rule_amortized, lr=0.05, max_steps=50, patience=5)
    
    print("\nApplying Hybrid TTT Ensemble to Query Set...")
    with torch.no_grad():
        top_k = z_rule_ttt.size(0)
        
        z = model.core.encode_context(q_obs_enc[:, 0].expand(top_k, -1), sample=False)["z"]
        memory = model.core.initial_memory(top_k, z.device)
        
        zs = []
        for t in range(T_full):
            step_dict = model.core.transition(z, torch.zeros(top_k, dtype=torch.long), memory, task_emb=z_rule_ttt)
            z = step_dict["z_next"]
            memory = step_dict["memory_next"]
            zs.append(z)
            
        zs = torch.stack(zs, dim=1)
        grid_logits = decoder(zs)
        
        preds_k = grid_logits.argmax(dim=2)
        preds, _ = torch.mode(preds_k, dim=0)
        
        acc_ttt = (preds.unsqueeze(0) == q_next_obs_grids).float().mean()
        print(f"Hybrid TTT Ensemble Accuracy: {acc_ttt.item()*100:.1f}%")

if __name__ == "__main__":
    run_scale_ttt()
