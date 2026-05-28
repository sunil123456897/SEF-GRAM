import torch
import torch.nn.functional as F
import numpy as np
import argparse

from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder
from sef_gram.task_encoder import TaskEncoder
from sef_gram.re_arc_loader import ReArcDataset

def run_latent_control_test(pretrain_steps=100):
    print("=== Latent Control Test ===")
    
    latent_dim = 64
    num_slots = 4
    cfg = WorldModelConfig(
        max_obs_dim=latent_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=11, 
        block_size_k=1,
        use_efla=True
    )
    
    model = UniversalWorldModel(cfg)
    encoder = ARCGridEncoder(latent_dim, vocab_size=11)
    decoder = ARCGridDecoder(latent_dim, vocab_size=11)
    task_encoder = TaskEncoder(latent_dim, num_slots=num_slots)
    
    dataset = ReArcDataset()
    
    params = list(model.parameters()) + list(encoder.parameters()) + list(task_encoder.parameters())
    optimizer_backbone = torch.optim.Adam(params, lr=0.001)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=0.005)
    
    print(f"Pretraining for {pretrain_steps} steps...")
    B = 4
    model.train()
    encoder.train()
    decoder.train()
    task_encoder.train()
    
    for ep in range(pretrain_steps):
        so, sn, qo, qn = dataset.get_batch(batch_size=B)
        obs = encoder(so)
        next_obs = encoder(sn)
        
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
        
        optimizer_dec.zero_grad()
        with torch.no_grad():
            obs_dec = encoder(so)
        grid_logits = decoder(obs_dec)
        ce_loss = F.cross_entropy(grid_logits.reshape(-1, 11, 30, 30), so.reshape(-1, 30, 30))
        ce_loss.backward()
        optimizer_dec.step()
        
    print("Pretraining complete. Running diagnostic...\n")
    
    model.eval()
    encoder.eval()
    decoder.eval()
    task_encoder.eval()
    
    so, sn, qo, qn = dataset.get_batch(batch_size=1)
    
    with torch.no_grad():
        s_obs_enc = encoder(so)
        s_target_enc = encoder(sn)
        q_obs_enc = encoder(qo)
        
        z_rule_amortized = task_encoder(s_obs_enc, s_target_enc) # [1, num_slots, latent_dim]
        
        z_rules = [z_rule_amortized]
        for _ in range(10):
            z_rules.append(torch.randn_like(z_rule_amortized))
            
        preds = []
        logits_list = []
        
        for z_rule in z_rules:
            # Query rollout
            z_q = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
            memory_q = model.core.initial_memory(1, z_q.device)
            step_dict_q = model.core.transition(z_q, torch.zeros(1, dtype=torch.long), memory_q, task_emb=z_rule[0:1])
            z_next_q = step_dict_q["z_next"].unsqueeze(1)
            
            grid_logits_q = decoder(z_next_q) # [1, 1, 11, 30, 30]
            pred_q = grid_logits_q.argmax(dim=2).squeeze(1).squeeze(0) # [30, 30]
            
            preds.append(pred_q)
            logits_list.append(grid_logits_q.squeeze(1).squeeze(0)) # [11, 30, 30]
            
    # Calculate Pairwise Pixel Diversity
    num_samples = len(preds)
    total_pairs = 0
    total_mismatch_rate = 0.0
    
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            mismatches = (preds[i] != preds[j]).float().mean().item()
            total_mismatch_rate += mismatches
            total_pairs += 1
            
    avg_mismatch_rate = total_mismatch_rate / total_pairs
    
    # Calculate Logit Variance
    logits_tensor = torch.stack(logits_list) # [11, 11, 30, 30]
    logit_variance = torch.var(logits_tensor, dim=0).mean().item()
    
    print(f"Total Rules Tested: {num_samples} (1 Amortized + 10 Random)")
    print(f"Average Pairwise Pixel Diversity: {avg_mismatch_rate * 100:.2f}%")
    print(f"Mean Logit Variance Across Rules: {logit_variance:.6f}\n")
    
    if avg_mismatch_rate < 0.03:
        print("DIAGNOSIS: FATAL BOTTLENECK 🚨")
        print("Decoder almost entirely ignores the latent z_rule. ")
        print("System-2 components (TTT, Verifier) will have NO causal effect on the output.")
    else:
        print("DIAGNOSIS: LATENT CONTROL ACTIVE ✅")
        print("Decoder actively changes predictions based on z_rule.")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_steps", type=int, default=100, help="Number of pretraining steps before diagnostic.")
    args = parser.parse_args()
    run_latent_control_test(args.pretrain_steps)
