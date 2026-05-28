import torch

class DirichletScheduler:
    """Safe Dirichlet Scheduler for GDPO Lambda.
    
    Prevents 'Collapse Deception' (where Dirichlet Energy goes to 0 due to Mode Collapse)
    by guarding the lambda transition with an Attractor Diversity check.
    """
    def __init__(self, diversity_threshold=0.15, temperature=10.0, score_threshold=0.6, ema_decay=0.85):
        self.diversity_threshold = diversity_threshold
        self.temperature = temperature
        self.score_threshold = score_threshold
        self.ema_decay = ema_decay
        
        self.max_dirichlet_history = 1e-6
        # Track a moving average of recent max to avoid being stuck at global max
        self.local_max_dirichlet = 1e-6
        self.lambda_gdpo = 0.0

    def step(self, current_dirichlet: float, active_attractors_count: float, vocab_size: int) -> float:
        if current_dirichlet > self.max_dirichlet_history:
            self.max_dirichlet_history = current_dirichlet
            
        # Update local max (smooth tracking)
        if current_dirichlet > self.local_max_dirichlet:
            self.local_max_dirichlet = current_dirichlet
        else:
            self.local_max_dirichlet = 0.99 * self.local_max_dirichlet + 0.01 * current_dirichlet
            
        dirichlet_score = 1.0 - (current_dirichlet / self.local_max_dirichlet)
        diversity_score = active_attractors_count / vocab_size
        
        # Guard: Only increase lambda if we have NOT collapsed the vocabulary
        if diversity_score > self.diversity_threshold:
            # Shifted sigmoid mapping [0, 1] score to [0, 1] lambda
            target_lambda = torch.sigmoid(torch.tensor(self.temperature * (dirichlet_score - self.score_threshold))).item()
        else:
            # Mode collapse detected! Lock RL to prevent catastrophic interference.
            target_lambda = 0.0
            
        # Smooth update
        self.lambda_gdpo = self.ema_decay * self.lambda_gdpo + (1 - self.ema_decay) * target_lambda
        return self.lambda_gdpo
