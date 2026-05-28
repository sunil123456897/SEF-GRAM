import torch

class Muon(torch.optim.Optimizer):
    """
    Оптимизатор Muon (Momentum Orthogonalized by Newton-Schulz) с Gram-ортогонализацией.
    Ортогонализирует обновления 2D-матриц весов.
    Экономит VRAM за счет отказа от хранения вторых моментов градиентов.
    Внедрена Gram-ортогонализация O(n^2) и численный стабилизатор 1e-7.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                # Инициализация буфера momentum
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                # Ортогонализация строго для 2D-матриц
                if p.data.ndim == 2:
                    X = buf.clone()
                    # Начальная нормировка по Фробениусу с повышенным эпсилоном
                    X /= (torch.norm(X) + 1e-7)
                    
                    rows, cols = X.shape
                    if rows >= cols:
                        # Gram Newton-Schulz через меньшую размерность cols x cols
                        for _ in range(ns_steps):
                            XTX = torch.matmul(X.t(), X)
                            eye = torch.eye(cols, device=X.device, dtype=X.dtype)
                            X = torch.matmul(X, 1.5 * eye - 0.5 * XTX)
                    else:
                        # Gram Newton-Schulz через меньшую размерность rows x rows
                        for _ in range(ns_steps):
                            XXT = torch.matmul(X, X.t())
                            eye = torch.eye(rows, device=X.device, dtype=X.dtype)
                            X = torch.matmul(1.5 * eye - 0.5 * XXT, X)
                    
                    # Стабилизация: предотвращаем появление NaN/Inf
                    if torch.isfinite(X).all():
                        p.data.add_(X, alpha=-lr)
                else:
                    # 1D параметры (векторы, biases) обновляются классическим momentum шагом только при конечных значениях
                    if torch.isfinite(buf).all():
                        p.data.add_(buf, alpha=-lr)

        return loss


class MuonWithAuxAdam:
    """
    Гибридный оптимизатор Muon с вспомогательным AdamW (MuonWithAuxAdam).
    - Для двумерных (2D) параметров использует ортогональный Muon.
    - Для одномерных (1D) параметров (нормы, biases, эмбеддинги) использует AdamW.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5, adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.01):
        params_list = list(params)
        
        muon_params = []
        adamw_params = []
        
        for p in params_list:
            if p.requires_grad:
                if p.ndim == 2:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)
                    
        self.muon_opt = Muon(muon_params, lr=lr, momentum=momentum, ns_steps=ns_steps) if muon_params else None
        self.adamw_opt = torch.optim.AdamW(adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_wd) if adamw_params else None

    def zero_grad(self):
        if self.muon_opt:
            self.muon_opt.zero_grad()
        if self.adamw_opt:
            self.adamw_opt.zero_grad()

    def step(self):
        if self.muon_opt:
            self.muon_opt.step()
        if self.adamw_opt:
            self.adamw_opt.step()
