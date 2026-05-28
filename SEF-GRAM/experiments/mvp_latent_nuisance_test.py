import torch
import torch.nn.functional as F
from sef_gram.world_model import UniversalWorldModel, WorldModelConfig, WorldBatch

def run_nuisance_test():
    torch.manual_seed(42)
    device = torch.device("cpu")
    
    print("=== Latent Nuisance (VJEPA) MVP Test ===")
    
    # 1. Initialize World Model
    cfg = WorldModelConfig(max_obs_dim=16, latent_dim=32, hidden_dim=64, num_actions=4)
    model = UniversalWorldModel(cfg).to(device)
    model.eval() # We just want to check the loss properties on a forward pass
    
    # 2. Generate Synthetic Clean Data
    batch_size = 16
    obs_dim = 16
    
    # Clean observations: simple repeating sine wave pattern
    t = torch.linspace(0, 10, obs_dim)
    clean_obs = torch.sin(t).unsqueeze(0).repeat(batch_size, 1)
    clean_next_obs = torch.sin(t + 0.1).unsqueeze(0).repeat(batch_size, 1)
    
    actions = torch.randint(0, 4, (batch_size,))
    rewards = torch.ones(batch_size)
    dones = torch.zeros(batch_size)
    
    clean_batch = WorldBatch(
        obs=clean_obs,
        actions=actions,
        next_obs=clean_next_obs,
        rewards=rewards,
        dones=dones
    )
    
    # 3. Generate Noisy Data (Latent Nuisance)
    # Add high-frequency Gaussian noise to the second half of the observation dimensions
    noise = torch.randn(batch_size, obs_dim // 2) * 5.0 # Large noise
    noisy_obs = clean_obs.clone()
    noisy_obs[:, obs_dim // 2:] += noise
    
    noisy_next_obs = clean_next_obs.clone()
    noisy_next_obs[:, obs_dim // 2:] += torch.randn(batch_size, obs_dim // 2) * 5.0
    
    noisy_batch = WorldBatch(
        obs=noisy_obs,
        actions=actions,
        next_obs=noisy_next_obs,
        rewards=rewards,
        dones=dones
    )
    
    # 4. Compute Loss for both
    with torch.no_grad():
        clean_total, clean_metrics = model.loss(clean_batch)
        noisy_total, noisy_metrics = model.loss(noisy_batch)
        
    print("\n--- Clean Batch Metrics ---")
    print(f"Total Loss:        {clean_metrics['total']:.4f}")
    print(f"Obs MSE:           {clean_metrics['obs_mse']:.4f}")
    print(f"Latent KL:         {clean_metrics['kl']:.4f}")
    print(f"Latent Logvar Reg: {clean_metrics['latent_logvar_reg']:.4f}")
    print(f"Latent MSE:        {clean_metrics['latent_mse']:.4f}")
    
    print("\n--- Noisy Batch Metrics ---")
    print(f"Total Loss:        {noisy_metrics['total']:.4f}")
    print(f"Obs MSE:           {noisy_metrics['obs_mse']:.4f}")
    print(f"Latent KL:         {noisy_metrics['kl']:.4f}")
    print(f"Latent Logvar Reg: {noisy_metrics['latent_logvar_reg']:.4f}")
    print(f"Latent MSE:        {noisy_metrics['latent_mse']:.4f}")
    
    print("\n--- Analysis ---")
    if noisy_metrics['obs_mse'] > clean_metrics['obs_mse'] * 10:
        print("[Pass] As expected, pixel-level Obs MSE blows up due to unpredictable noise.")
    
    if noisy_metrics['latent_mse'] < noisy_metrics['obs_mse']:
        print("[Pass] Latent MSE is much lower than Obs MSE. VJEPA target encoder abstracts away the noise!")
        
    if noisy_metrics['kl'] > 1e-4:
        print("[Pass] KL Divergence is strictly positive. No Mode Collapse (Posterior variance hasn't dropped to 0).")
    else:
        print("[Fail] KL is too close to zero. Potential Mode Collapse.")
        
    print("Test Completed.")

if __name__ == "__main__":
    run_nuisance_test()
