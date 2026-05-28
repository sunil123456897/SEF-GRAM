import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import json
import os
import copy
import argparse

from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder
from sef_gram.task_encoder import TaskEncoder
from sef_gram.ttt_planner import HybridTTTPlanner
from sef_gram.re_arc_loader import ReArcDataset

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def evaluate_mode(mode_name, config_flags, seed, eval_tasks=5, pretrain_steps=20):
    set_seed(seed)
    
    latent_dim = 64
    num_slots = 4
    cfg = WorldModelConfig(
        max_obs_dim=latent_dim,
        latent_dim=latent_dim,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=11, 
        block_size_k=1,
        use_efla=config_flags.get("use_efla", True)
    )
    
    # Mode overrides
    use_efla = config_flags.get("use_efla", True)
    use_ttt = config_flags.get("use_ttt", True)
    use_verifier = config_flags.get("use_verifier", True)
    use_warm_start = config_flags.get("use_warm_start", True)
    use_decoder_ttft = config_flags.get("use_decoder_ttft", True)
    
    # ---------------------------------------------------------------------
    # 1. Pretraining
    # ---------------------------------------------------------------------
    model = UniversalWorldModel(cfg)
    
    encoder = ARCGridEncoder(latent_dim, vocab_size=11)
    decoder = ARCGridDecoder(latent_dim, vocab_size=11)
    task_encoder = TaskEncoder(latent_dim, num_slots=num_slots)
    
    dataset = ReArcDataset()
    
    params = list(model.parameters()) + list(encoder.parameters()) + list(task_encoder.parameters())
    optimizer_backbone = torch.optim.Adam(params, lr=0.001)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=0.005)
    
    B = 4
    final_ce_loss = 0.0
    for ep in range(pretrain_steps):
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
        final_ce_loss = ce_loss.item()
        
    # ---------------------------------------------------------------------
    # 2. Evaluation
    # ---------------------------------------------------------------------
    for p in model.parameters(): p.requires_grad = False
    for p in encoder.parameters(): p.requires_grad = False
    for p in task_encoder.parameters(): p.requires_grad = False
    
    model.eval()
    encoder.eval()
    task_encoder.eval()
    
    exact_match_count = 0
    total_query_pixel_accuracy = 0.0
    total_query_pixel_error = 0.0
    total_support_best_error = 0.0
    
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
            if use_warm_start:
                z_rule_amortized = task_encoder(s_obs_enc, s_target_enc)
            else:
                z_rule_amortized = torch.randn(1, num_slots, latent_dim, device=s_obs_enc.device)
            
        planner = HybridTTTPlanner(model, latent_dim, num_slots=num_slots, num_hypotheses=16, top_k=4)
        s_obs_enc_sq = s_obs_enc.squeeze(0)
        s_target_enc_sq = s_target_enc.squeeze(0).unsqueeze(1)
        
        if use_ttt:
            z_rule_ttt = planner.plan(s_obs_enc_sq, s_target_enc_sq, T=1, z_amortized=z_rule_amortized, lr=0.05, max_steps=10, patience=2)
        else:
            z_rule_ttt = z_rule_amortized # [1, num_slots, dim]
            
        final_z_rule = z_rule_ttt[0:1] # Default (w/o Support Verifier), [1, num_slots, dim]
        best_error = float('inf') # Default
        
        if use_verifier:
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
            
        if use_decoder_ttft:
            opt_decoder_tt = torch.optim.Adam(decoder.parameters(), lr=0.01)
            z_cand_exp = final_z_rule.expand(K_supp, -1, -1)
            
            with torch.no_grad():
                z = model.core.encode_context(s_obs_enc_sq, sample=False)["z"]
                memory = model.core.initial_memory(K_supp, z.device)
                step_dict = model.core.transition(z, torch.zeros(K_supp, dtype=torch.long), memory, task_emb=z_cand_exp)
                z_next = step_dict["z_next"].unsqueeze(1)
                target_pixels = sn.squeeze(0)
                
            for tt_step in range(5):
                opt_decoder_tt.zero_grad()
                grid_logits = decoder(z_next)
                loss_tt = F.cross_entropy(grid_logits.reshape(-1, 11, 30, 30), target_pixels.reshape(-1, 30, 30))
                loss_tt.backward()
                opt_decoder_tt.step()
                
        # Query Evaluation
        decoder.eval()
        with torch.no_grad():
            z_q = model.core.encode_context(q_obs_enc[:, 0], sample=False)["z"]
            memory_q = model.core.initial_memory(1, z_q.device)
            step_dict_q = model.core.transition(z_q, torch.zeros(1, dtype=torch.long), memory_q, task_emb=final_z_rule)
            z_next_q = step_dict_q["z_next"].unsqueeze(1)
            
            grid_logits_q = decoder(z_next_q)
            pred_q = grid_logits_q.argmax(dim=2).squeeze(1).squeeze(0)
            target_q = qn.squeeze(0).squeeze(0)
            
            pixels_correct = (pred_q == target_q).sum().item()
            total_pixels = target_q.numel()
            pixels_wrong = total_pixels - pixels_correct
            
            total_query_pixel_error += pixels_wrong
            total_query_pixel_accuracy += (pixels_correct / total_pixels) * 100
            
            if pixels_wrong == 0:
                exact_match_count += 1
                
        total_support_best_error += best_error if best_error != float('inf') else 900
                
        decoder.load_state_dict(decoder_state)

    success_rate = (exact_match_count / eval_tasks) * 100
    mean_query_pixel_accuracy = total_query_pixel_accuracy / eval_tasks
    mean_query_pixel_error = total_query_pixel_error / eval_tasks
    mean_support_best_error = total_support_best_error / eval_tasks
    
    return {
        "final_ce_loss": final_ce_loss,
        "success_rate": success_rate,
        "query_pixel_accuracy": mean_query_pixel_accuracy,
        "query_pixel_error": mean_query_pixel_error,
        "support_best_error": mean_support_best_error
    }

def run_all_ablations(args):
    print("=== Phase 7: Scientific Ablation Studies ===")
    
    modes = [
        ("Full SEF-GRAM", {"use_efla": True, "use_ttt": True, "use_verifier": True, "use_warm_start": True, "use_decoder_ttft": True}),
        ("w/o EFLA", {"use_efla": False, "use_ttt": True, "use_verifier": True, "use_warm_start": True, "use_decoder_ttft": True}),
        ("w/o TTT Planner", {"use_efla": True, "use_ttt": False, "use_verifier": True, "use_warm_start": True, "use_decoder_ttft": True}),
        ("w/o Support Verifier", {"use_efla": True, "use_ttt": True, "use_verifier": False, "use_warm_start": True, "use_decoder_ttft": True}),
        ("w/o Task Encoder warm-start", {"use_efla": True, "use_ttt": True, "use_verifier": True, "use_warm_start": False, "use_decoder_ttft": True}),
        ("w/o Decoder TTFT", {"use_efla": True, "use_ttt": True, "use_verifier": True, "use_warm_start": True, "use_decoder_ttft": False}),
    ]
    
    seeds = [42, 100, 2026]
    results = []
    
    for mode_name, flags in modes:
        print(f"\nRunning ablation: {mode_name}...")
        metrics = {
            "ce_losses": [],
            "success_rates": [],
            "pixel_accs": [],
            "pixel_errs": [],
            "support_errs": []
        }
        
        for seed in seeds:
            print(f"  Seed {seed}...")
            res = evaluate_mode(mode_name, flags, seed, eval_tasks=args.eval_tasks, pretrain_steps=args.pretrain_steps)
            metrics["ce_losses"].append(res["final_ce_loss"])
            metrics["success_rates"].append(res["success_rate"])
            metrics["pixel_accs"].append(res["query_pixel_accuracy"])
            metrics["pixel_errs"].append(res["query_pixel_error"])
            metrics["support_errs"].append(res["support_best_error"])
            
        results.append({
            "Mode": mode_name,
            "Decoder CE": np.mean(metrics["ce_losses"]),
            "Success Rate (%)": np.mean(metrics["success_rates"]),
            "Query Pixel Acc (%)": np.mean(metrics["pixel_accs"]),
            "Query Pixel Err": np.mean(metrics["pixel_errs"]),
            "Support Best Err": np.mean(metrics["support_errs"])
        })
        
    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/phase7_ablations.csv", index=False)
    
    # Save JSON
    with open("results/phase7_ablations.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Save Markdown
    os.makedirs("docs", exist_ok=True)
    md_str = df.to_markdown(index=False, floatfmt=".2f")
    with open("docs/phase7_ablation_results.md", "w") as f:
        f.write("# Phase 7: Component Ablations\n\n")
        f.write(md_str)
        f.write("\n")
        
    print("\n=== Final Results ===")
    print(md_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_tasks", type=int, default=5, help="Number of tasks for evaluation.")
    parser.add_argument("--pretrain_steps", type=int, default=20, help="Number of pretraining steps.")
    args = parser.parse_args()
    run_all_ablations(args)
