import torch
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch
from sef_gram.arc_dataset import ARCGridEncoder, ARCGridDecoder, generate_flood_fill_trajectory

def test_arc_overfit():
    print("=== Phase 4.2: ARC Sanity Check (Flood Fill Overfitting) ===")
    
    latent_dim = 64
    cfg = WorldModelConfig(
        max_obs_dim=latent_dim, # We feed CNN embeddings as obs
        latent_dim=latent_dim,
        hidden_dim=128,
        num_actions=4,
        env_vocab_size=10, 
        block_size_k=4
    )
    
    model = UniversalWorldModel(cfg)
    encoder = ARCGridEncoder(latent_dim)
    decoder = ARCGridDecoder(latent_dim)
    
    # Optimizer for all components
    params = list(model.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    B, T_full = 1, 10
    # Generate 1 sequence of Flood Fill (T=11 steps total -> T=10 transitions)
    grids = generate_flood_fill_trajectory(B, T_full+1)
    obs_grids = grids[:, :-1]
    next_obs_grids = grids[:, 1:]
    
    print("Starting Training (Overfitting on 1 ARC Sequence for 300 iterations)...")
    
    for i in range(300):
        optimizer.zero_grad()
        
        # 1. Encode Grids
        obs = encoder(obs_grids)           # [1, 10, 64]
        next_obs = encoder(next_obs_grids) # [1, 10, 64]
        
        batch = WorldBatch(
            obs=obs,
            actions=torch.zeros(B, T_full, dtype=torch.long),
            next_obs=next_obs,
            rewards=torch.zeros(B, T_full),
            dones=torch.zeros(B, T_full)
        )
        
        # 2. SEF-GRAM Forward & VJEPA Loss
        loss, metrics = model.loss(batch, lambda_gdpo=0.0)
        
        # 3. Grid Reconstruction Loss
        # Extract predictions
        out = model.forward(batch)
        z_preds = out["z_preds"] # [B, K, 64]
        
        # Decode predicted latents to grid logits
        grid_logits = decoder(z_preds) # [B, K, 10, 30, 30]
        
        # Target grids for the active block
        t_start = out["t_start"]
        K = out["K"]
        target_grids = next_obs_grids[:, t_start:t_start+K] # [B, K, 30, 30]
        
        ce_loss = F.cross_entropy(grid_logits.reshape(-1, 10, 30, 30), target_grids.reshape(-1, 30, 30))
        
        total_loss = loss + ce_loss
        total_loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            preds = grid_logits.argmax(dim=2)
            acc = (preds == target_grids).float().mean()
            print(f"Iter {i+1:3d} | VJEPA Loss: {metrics['echo_loss'].item():.4f} | Grid CE Loss: {ce_loss.item():.4f} | Pixel Acc: {acc.item()*100:.1f}%")

    print("\n[VERIFICATION] Generating Rollout to inspect latent spatial reasoning...")
    with torch.no_grad():
        # Encode initial state t=0
        z_0_enc = encoder(obs_grids[:, 0:1])
        # Start rollout
        enc_ctx = model.core.encode_context(z_0_enc[:, 0], sample=False)
        z = enc_ctx["z"]
        memory = model.core.initial_memory(B, z.device)
        
        rollout_zs = []
        for t in range(5): # Check next 5 steps
            step = model.core.transition(z, torch.zeros(B, dtype=torch.long), memory)
            z = step["z_next"]
            memory = step["memory_next"]
            rollout_zs.append(z)
            
        rollout_zs = torch.stack(rollout_zs, dim=1)
        grid_logits = decoder(rollout_zs)
        preds = grid_logits.argmax(dim=2)
        
        # Compare prediction at step 3 with ground truth step 3
        acc_step3 = (preds[:, 2] == next_obs_grids[:, 2]).float().mean()
        print(f"Rollout Acc (Step 3 prediction vs Target): {acc_step3.item()*100:.1f}%")
        if acc_step3.item() > 0.95:
            print("[SUCCESS] SEF-GRAM successfully learned to execute the Flood Fill algorithm step-by-step purely in latent space!")
        else:
            print("[WARNING] Pixel accuracy is below 95%. Algorithm may need more training.")

if __name__ == "__main__":
    test_arc_overfit()
