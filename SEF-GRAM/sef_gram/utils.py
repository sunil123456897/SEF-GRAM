import torch

def taylor_stabilized_gate(k, alpha_scale, eps=1e-8):
    """
    Вычисляет Тейлор-стабилизированный гейт для EFLA ODE шага.
    k: Key-тензор формы (B, D)
    alpha_scale: Обучаемый масштабный коэффициент
    eps: Эпсилон для предотвращения деления на ноль
    """
    # Вычисляем lambda = ||k||^2 для каждого батча
    k_len_sq = torch.sum(k ** 2, dim=-1, keepdim=True)  # (B, 1)
    beta = torch.sigmoid(alpha_scale)  # Базовый коэффициент
    x = beta * k_len_sq  # (B, 1)
    
    # Taylor-аппроксимация при малых x < 1e-5 для предотвращения 0/0 на fp16/bf16
    factor = torch.where(
        x < 1e-5,
        1.0 - 0.5 * x + (1.0 / 6.0) * (x ** 2),
        -torch.expm1(-x) / (x + eps)
    )
    return beta * factor  # (B, 1)


def compute_dirichlet_energy(h, adjacency=None):
    """
    Вычисляет многомерную энергию Дирихле для латентных векторов h формы (B, D).
    Используется для детекции реляционных аналогий ("Ага!-момента").
    E = sum_{i,j} A_{i,j} ||h_i - h_j||^2
    Если матрица смежности adjacency (B, B) не задана, считается полносвязный граф.
    """
    B, D = h.shape
    if B <= 1:
        return torch.tensor(0.0, device=h.device)
        
    # Попарные евклидовы расстояния в квадрате
    dists_sq = torch.cdist(h, h, p=2) ** 2  # (B, B)
    
    if adjacency is None:
        # Полносвязный граф без самопетлей с весом 1 / (B - 1)
        adjacency = torch.ones(B, B, device=h.device) - torch.eye(B, device=h.device)
        adjacency = adjacency / (B - 1)
        
    energy = 0.5 * torch.sum(adjacency * dists_sq)
    return energy
