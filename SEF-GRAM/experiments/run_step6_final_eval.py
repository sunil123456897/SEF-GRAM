import torch
import torch.nn as nn
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder
from sef_gram.task_encoder import TaskEncoder
from sef_gram.ttt_planner import HybridTTTPlanner
from sef_gram.re_arc_loader import ReArcDataset

def run_phase_6():
    print("=== Phase 6: System 2 Verification (Final Leaderboard Eval) ===")
    
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
    
    # ---------------------------------------------------------------------
    # 1. Train Backbone & Decoder
    # ---------------------------------------------------------------------
    print("\n[Phase 5.3 + 5.4 Simulation] Pretraining SEF-GRAM...")
    params = list(model.parameters()) + list(encoder.parameters()) + list(task_encoder.parameters())
    optimizer_backbone = torch.optim.Adam(params, lr=0.001)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=0.005)
    
    B = 4
    for ep in range(150):
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
        
        optimizer_dec.zero_grad()
        with torch.no_grad():
            obs_dec = encoder(so)
        grid_logits = decoder(obs_dec)
        ce_loss = F.cross_entropy(grid_logits.reshape(-1, 11, 30, 30), so.reshape(-1, 30, 30))
        ce_loss.backward()
        optimizer_dec.step()

    print(f"Pretraining Complete. VJEPA Loss: {loss.item():.4f} | Decoder CE: {ce_loss.item():.4f}")
    
    # ---------------------------------------------------------------------
    # 2. Phase 6 Evaluation
    # ---------------------------------------------------------------------
    print("\n[EVALUATION] Running Leaderboard Evaluation on 20 Random Tasks...")
    
    for p in model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False
    for p in task_encoder.parameters(): p.requires_grad = False
    
    model.eval()
    encoder.eval()
    task_encoder.eval()
    
    eval_tasks = 20
    exact_match_count = 0
    
    for task_idx in range(eval_tasks):
        decoder_state = {k: v.clone() for k, v in decoder.state_dict().items()}
        decoder.train()
        for p in decoder.parameters(): p.requires_grad = True
        
        so, sn, qo, qn = dataset.get_batch(batch_size=1)
        K_supp = so.shape[1]
        
        with torch.no_grad():
            s_obs_enc = encoder(so)
            s_target_enc = encoder(sn)
            q_obs_enc = encoder(qo)
            z_rule_amortized = task_encoder(s_obs_enc, s_target_enc)
            
        planner = HybridTTTPlanner(model, latent_dim, num_slots=num_slots, num_hypotheses=32, top_k=10)
        s_obs_enc_sq = s_obs_enc.squeeze(0)
        s_target_enc_sq = s_target_enc.squeeze(0).unsqueeze(1)
        
        z_rule_ttt = planner.plan(s_obs_enc_sq, s_target_enc_sq, T=1, z_amortized=z_rule_amortized, lr=0.05, max_steps=30, patience=5)
        
        # Pixel-Level Support Verification
        valid_z_rules = []
        best_error = float('inf')
        best_z_rule = None
        
        top_k_candidates = z_rule_ttt.size(0)
        for i in range(top_k_candidates):
            z_cand = z_rule_ttt[i:i+1]
            z_cand_exp = z_cand.expand(K_supp, -1, -1)
            
            with torch.no_grad():
                z = model.core.encode_context(s_obs_enc_sq, sample=False)["z"]
                memory = model.core.initial_memory(K_supp, z.device)
                step_dict = model.core.transition(z, torch.zeros(K_supp, dtype=torch.long), memory, task_emb=z_cand_exp)
                z_next = step_dict["z_next"]
                
                grid_logits = decoder(z_next.unsqueeze(1))
                preds = grid_logits.argmax(dim=2).squeeze(1)
                target_pixels = sn.squeeze(0)
                
                errors = (preds != target_pixels).sum().item()
                
            if errors == 0:
                valid_z_rules.append(z_cand)
            
            if errors < best_error:
                best_error = errors
                best_z_rule = z_cand
                
        final_z_rule = valid_z_rules[0] if len(valid_z_rules) > 0 else best_z_rule
        
        # Test-Time Fine-Tuning of Decoder
        opt_decoder_tt = torch.optim.Adam(decoder.parameters(), lr=0.01)
        z_cand_exp = final_z_rule.expand(K_supp, -1, -1)
        
        with torch.no_grad():
            z = model.core.encode_context(s_obs_enc_sq, sample=False)["z"]
            memory = model.core.initial_memory(K_supp, z.device)
            step_dict = model.core.transition(z, torch.zeros(K_supp, dtype=torch.long), memory, task_emb=z_cand_exp)
            z_next = step_dict["z_next"].unsqueeze(1)
            target_pixels = sn.squeeze(0)
            
        for tt_step in range(10):
            opt_decoder_tt.zero_grad()
            grid_logits = decoder(z_next)
            loss_tt = F.cross_entropy(grid_logits.reshape(-1, 11, 30, 30), target_pixels.reshape(-1, 30, 30))
            loss_tt.backward()
            opt_decoder_tt.step()
            
        # Final Query Evaluation
        decoder.eval()
        with torch.no_grad():
            z_q = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
            memory_q = model.core.initial_memory(1, z_q.device)
            step_dict_q = model.core.transition(z_q, torch.zeros(1, dtype=torch.long), memory_q, task_emb=final_z_rule)
            z_next_q = step_dict_q["z_next"].unsqueeze(1)
            
            grid_logits_q = decoder(z_next_q)
            pred_q = grid_logits_q.argmax(dim=2).squeeze(1).squeeze(0)
            target_q = qn.squeeze(0).squeeze(0)
            
            if torch.all(pred_q == target_q):
                exact_match_count += 1
                match_str = "SUCCESS"
            else:
                match_str = "FAILED "
                
        print(f"Task {task_idx+1:02d}/20 | Verification: {'Exact' if len(valid_z_rules)>0 else f'{best_error} errs'} | Result: {match_str}")
        
        # Restore decoder weights
        decoder.load_state_dict(decoder_state)

    success_rate = (exact_match_count / eval_tasks) * 100
    print(f"\nFinal Leaderboard Evaluation: {exact_match_count}/{eval_tasks} Exact Matches")
    print(f"Task Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    run_phase_6()
