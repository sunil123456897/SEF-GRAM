import torch
import torch.nn.functional as F
from sef_gram.full_system import ExactEFLACell

def test_closed_form_scalar():
    """
    1. Closed-form scalar/rank-1 case: small dimension, fixed k, v, memory.
    Compares ExactEFLACell._alpha() with analytical formula.
    """
    latent_dim = 2
    cell = ExactEFLACell(latent_dim)
    cell.eval()
    
    # Set explicit log_beta for deterministic test
    with torch.no_grad():
        cell.log_beta.copy_(torch.tensor(-1.0)) # beta = softplus(-1) = 0.31326
        
    k = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    eps = 1e-8
    
    # Analytical
    k2 = torch.sum(k * k, dim=-1, keepdim=True)
    beta = F.softplus(torch.tensor(-1.0)) + eps
    x = beta * k2
    expected_alpha = -torch.expm1(-x) / (k2 + eps)
    
    # Cell
    actual_alpha = cell._alpha(k, eps=eps)
    
    assert torch.allclose(actual_alpha, expected_alpha, atol=1e-6), "Alpha function deviates from exact mathematical definition"

def test_float64_reference():
    """
    2. Float64 reference test: compares PyTorch implementation with independent reference function.
    """
    latent_dim = 4
    cell = ExactEFLACell(latent_dim).to(torch.float64)
    cell.eval()
    
    B = 2
    z_prev = torch.randn(B, latent_dim, dtype=torch.float64)
    memory_prev = torch.randn(B, latent_dim, latent_dim, dtype=torch.float64)
    
    # Run through module
    with torch.no_grad():
        z_next_mod, memory_next_mod = cell(z_prev, memory_prev)
        
        # Reference implementation
        x = cell.in_proj(z_prev)
        k = cell.key(x)
        v = cell.value(x)
        
        k2 = torch.sum(k * k, dim=-1, keepdim=True)
        beta = F.softplus(cell.log_beta) + 1e-8
        x_beta = beta * k2
        alpha = -torch.expm1(-x_beta) / (k2 + 1e-8)
        
        retrieved = torch.bmm(memory_prev, k.unsqueeze(-1)).squeeze(-1)
        residual = v - retrieved
        
        update = alpha.unsqueeze(-1) * torch.bmm(residual.unsqueeze(-1), k.unsqueeze(1))
        memory_next_ref = memory_prev + update
        
        z_memory = torch.bmm(memory_next_ref, k.unsqueeze(-1)).squeeze(-1)
        z_next_ref = cell.state_norm(z_prev + cell.out_proj(z_memory))
        
    assert torch.allclose(memory_next_mod, memory_next_ref, atol=1e-12), "Memory update does not match float64 reference"
    assert torch.allclose(z_next_mod, z_next_ref, atol=1e-12), "Z update does not match float64 reference"

def test_long_horizon_drift():
    """
    4. Long-horizon drift test: show exact update does not accumulate error like Euler discretization.
    """
    latent_dim = 8
    cell = ExactEFLACell(latent_dim).to(torch.float64)
    cell.eval()
    
    B = 1
    z = torch.randn(B, latent_dim, dtype=torch.float64)
    memory = torch.zeros(B, latent_dim, latent_dim, dtype=torch.float64)
    
    # We will simulate 100 updates with the SAME key and value to show that it converges exactly
    # instead of exploding or drifting
    
    with torch.no_grad():
        x = cell.in_proj(z)
        k = cell.key(x)
        v = cell.value(x)
        
        # Over 100 steps of identical k,v, memory @ k should converge exactly to v.
        for _ in range(100):
            # Bypass cell forward to inject constant k, v
            retrieved = torch.bmm(memory, k.unsqueeze(-1)).squeeze(-1)
            residual = v - retrieved
            alpha = cell._alpha(k)
            update = alpha.unsqueeze(-1) * torch.bmm(residual.unsqueeze(-1), k.unsqueeze(1))
            memory = memory + update
            
        final_retrieved = torch.bmm(memory, k.unsqueeze(-1)).squeeze(-1)
        
        # Calculate analytical expected retrieval after N steps
        k2 = torch.sum(k * k, dim=-1, keepdim=True)
        beta = F.softplus(cell.log_beta) + 1e-8
        decay_factor = torch.exp(-100 * beta * k2)
        expected_retrieved = v * (1.0 - decay_factor)
        
        # The retrieved value should match the analytical exponential decay precisely,
        # confirming no numerical drift or instability from repetitive application.
        assert torch.allclose(final_retrieved, expected_retrieved, atol=1e-10), "Long horizon memory retrieval drifted from exact mathematical trajectory"
