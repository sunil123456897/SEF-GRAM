import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os

from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder
from sef_gram.task_encoder import TaskEncoder
from sef_gram.re_arc_loader import ReArcDataset

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate_latent_control(model, encoder, decoder, task_encoder, dataset):
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
            step_dict_q = model.core.transition(z_q, torch.zeros(1, dtype=torch.long), memory_q, task_emb=z_rule)
            z_next_q = step_dict_q["z_next"].unsqueeze(1)
            
            grid_logits_q = decoder(z_next_q, z_rule=z_rule) # [1, 1, 11, 30, 30]
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
    
    model.train()
    encoder.train()
    decoder.train()
    task_encoder.train()
    
    return avg_mismatch_rate, logit_variance

def run_training(steps=1000, margin=1.0):
    print("=== Phase 8.0: Causal Rule Grounding (Concat + Contrastive) ===")
    
    set_seed(42)
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
    
    params = list(model.parameters()) + list(encoder.parameters()) + list(task_encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    B = 8
    model.train()
    encoder.train()
    decoder.train()
    task_encoder.train()
    
    for step in range(1, steps + 1):
        so, sn, qo, qn = dataset.get_batch(batch_size=B)
        
        # 1. Encode
        s_obs_enc = encoder(so)
        s_target_enc = encoder(sn)
        q_obs_enc = encoder(qo)
        
        # 2. Get Rule
        z_rule_correct = task_encoder(s_obs_enc, s_target_enc)
        
        # 3. Transition correct rule
        z_q = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
        memory_q = model.core.initial_memory(B, z_q.device)
        step_correct = model.core.transition(z_q, torch.zeros(B, dtype=torch.long), memory_q, task_emb=z_rule_correct)
        z_next_correct = step_correct["z_next"].unsqueeze(1)
        
        # 4. Decode correct rule
        logits_correct = decoder(z_next_correct, z_rule=z_rule_correct)
        loss_correct = F.cross_entropy(logits_correct.reshape(-1, 11, 30, 30), qn.reshape(-1, 30, 30))
        
        # 5. Wrong rule (shifted batch)
        z_rule_wrong = torch.roll(z_rule_correct, shifts=1, dims=0)
        
        # 6. Transition wrong rule
        step_wrong = model.core.transition(z_q, torch.zeros(B, dtype=torch.long), memory_q, task_emb=z_rule_wrong)
        z_next_wrong = step_wrong["z_next"].unsqueeze(1)
        
        # 7. Decode wrong rule
        logits_wrong = decoder(z_next_wrong, z_rule=z_rule_wrong)
        loss_wrong = F.cross_entropy(logits_wrong.reshape(-1, 11, 30, 30), qn.reshape(-1, 30, 30))
        
        # 8. Contrastive Margin Loss
        contrastive_loss = F.relu(margin + loss_correct - loss_wrong)
        total_loss = loss_correct + contrastive_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                pred_correct = logits_correct.argmax(dim=2)
                acc_correct = (pred_correct == qn).float().mean().item() * 100
                pred_wrong = logits_wrong.argmax(dim=2)
                acc_wrong = (pred_wrong == qn).float().mean().item() * 100
            
            div, var = evaluate_latent_control(model, encoder, decoder, task_encoder, dataset)
            
            print(f"Step {step:04d} | L_corr: {loss_correct.item():.3f} | L_wrong: {loss_wrong.item():.3f} | "
                  f"Gap: {(loss_wrong - loss_correct).item():.3f} | ContrL: {contrastive_loss.item():.3f} | "
                  f"Acc_C: {acc_correct:.1f}% | Acc_W: {acc_wrong:.1f}% | "
                  f"Div: {div*100:.2f}% | Var: {var:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()
    run_training(args.steps, args.margin)
