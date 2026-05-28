import torch
import torch.nn as nn
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder
from sef_gram.task_encoder import TaskEncoder
from sef_gram.ttt_planner import HybridTTTPlanner
from sef_gram.re_arc_loader import ReArcDataset

def run_phase_5_4():
    print("=== Phase 5.4: Decoder Calibration & Linear Probing ===")
    
    latent_dim = 64
    num_slots = 4
    cfg = WorldModelConfig(
        max_obs_dim=latent_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=11, 
        block_size_k=1
    )
    
    model = UniversalWorldModel(cfg)
    encoder = ARCGridEncoder(latent_dim, vocab_size=11)
    decoder = ARCGridDecoder(latent_dim, vocab_size=11)
    task_encoder = TaskEncoder(latent_dim, num_slots=num_slots)
    
    dataset = ReArcDataset()
    
    print("\n[Phase 5.3 Simulation] Training VJEPA Backbone...")
    params = list(model.parameters()) + list(encoder.parameters()) + list(task_encoder.parameters())
    optimizer_backbone = torch.optim.Adam(params, lr=0.001)
    
    B = 4
    for ep in range(50):
        so, sn, qo, qn = dataset.get_batch(batch_size=B)
        obs = encoder(so)
        next_obs = encoder(sn)
        
        task_encoder.train()
        z_rule = task_encoder(obs, next_obs)
        
        batch = WorldBatch(
            obs=obs,
            actions=torch.zeros(B, 3, dtype=torch.long),
            next_obs=next_obs,
            rewards=torch.zeros(B, 3),
            dones=torch.zeros(B, 3),
            task_emb=z_rule
        )
        
        optimizer_backbone.zero_grad()
        loss, _ = model.loss(batch, lambda_gdpo=0.0)
        loss.backward()
        optimizer_backbone.step()
        
    print(f"Backbone Training Complete. Final VJEPA Loss: {loss.item():.4f}")
    
    print("\n[Phase 5.4] Freezing Backbone & Training Pixel Translator...")
    for p in model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False
    for p in task_encoder.parameters(): p.requires_grad = False
    
    model.eval()
    encoder.eval()
    task_encoder.eval()
    
    decoder.train()
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=0.005)
    
    for ep in range(100):
        so, sn, qo, qn = dataset.get_batch(batch_size=B)
        with torch.no_grad():
            obs = encoder(so) # [B, 3, 64]
            
        target_grids = so # [B, 3, 30, 30]
        
        optimizer_dec.zero_grad()
        grid_logits = decoder(obs) # [B, 3, 11, 30, 30]
        
        ce_loss = F.cross_entropy(grid_logits.reshape(-1, 11, 30, 30), target_grids.reshape(-1, 30, 30))
        ce_loss.backward()
        optimizer_dec.step()
        
        if (ep+1) % 20 == 0:
            preds = grid_logits.argmax(dim=2)
            acc = (preds == target_grids).float().mean()
            print(f"Decoder Epoch {ep+1:3d} | CE Loss: {ce_loss.item():.4f} | Train Acc: {acc.item()*100:.1f}%")
            
    print("Decoder Calibration Complete.")
    
    print("\n[EVALUATION] Testing on holdout ARC tasks with Hybrid TTT...")
    decoder.eval()
    
    so, sn, qo, qn = dataset.get_batch(batch_size=1)
    
    with torch.no_grad():
        s_obs_enc = encoder(so)
        s_target_enc = encoder(sn)
        q_obs_enc = encoder(qo)
        q_target_enc = encoder(qn) 
        
        z_rule_amortized = task_encoder(s_obs_enc, s_target_enc)
        
        z = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
        memory = model.core.initial_memory(1, z.device)
        step_dict = model.core.transition(z, torch.zeros(1, dtype=torch.long), memory, task_emb=z_rule_amortized)
        z_next_baseline = step_dict["z_next"]
        
        latent_loss_baseline = F.mse_loss(z_next_baseline, q_target_enc[:, 0])
        print(f"\nQuery Latent Loss (Task Encoder only): {latent_loss_baseline.item():.4f}")
        
    planner = HybridTTTPlanner(model, latent_dim, num_slots=num_slots, num_hypotheses=64, top_k=3)
    s_obs_enc_sq = s_obs_enc.squeeze(0)
    s_target_enc_sq = s_target_enc.squeeze(0).unsqueeze(1)
    
    z_rule_ttt = planner.plan(s_obs_enc_sq, s_target_enc_sq, T=1, z_amortized=z_rule_amortized, lr=0.05, max_steps=50, patience=5)
    
    with torch.no_grad():
        top_k = z_rule_ttt.size(0)
        
        z = model.core.encode_context(q_obs_enc[:, 0].expand(top_k, -1), sample=False)["z"]
        memory = model.core.initial_memory(top_k, z.device)
        
        step_dict = model.core.transition(z, torch.zeros(top_k, dtype=torch.long), memory, task_emb=z_rule_ttt)
        z_next_ttt = step_dict["z_next"]
        
        latent_loss_ttt = F.mse_loss(z_next_ttt[0].unsqueeze(0), q_target_enc[:, 0])
        print(f"Query Latent Loss (After Hybrid TTT Top-1): {latent_loss_ttt.item():.4f}")
        
        zs = z_next_ttt.unsqueeze(1)
        grid_logits = decoder(zs)
        preds = grid_logits.argmax(dim=2).squeeze(1)
        target = qn.squeeze(0).squeeze(0)
        
        accs = []
        for k in range(top_k):
            acc = (preds[k] == target).float().mean()
            accs.append(acc.item())
            print(f"Candidate {k+1} Pixel Accuracy: {acc.item()*100:.1f}%")
            
        print(f"\nFinal Top-3 Ensemble Maximum Pixel Accuracy: {max(accs)*100:.1f}%")

if __name__ == "__main__":
    run_phase_5_4()
