import torch
import torch.nn as nn
from sef_gram.utils import taylor_stabilized_gate

class VJEPAEncoder(nn.Module):
    """
    Вероятностный кодировщик VJEPA (Stochastic Latent Context Encoder).
    Кодирует входные данные в параметры диагональной гауссианы в латентном пространстве.
    """
    def __init__(self, input_dim=64, latent_dim=512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.fc_shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        """
        Прямой проход кодировщика.
        x: тензор входных признаков формы (B, input_dim)
        Возвращает:
            mu: среднее значение латентного вектора (B, latent_dim)
            logvar: логарифм дисперсии латентного вектора (B, latent_dim)
        """
        h = self.fc_shared(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def sample(self, mu, logvar):
        """
        Репараметризационный трюк (Reparameterization trick).
        Сэмплирует вектор Z_t из распределения N(mu, exp(logvar)).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class EFLACell(nn.Module):
    """
    Error-Free Linear Attention (EFLA) Cell.
    Реализует точный непрерывный переход латентных состояний в виде ODE ранга-1.
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.w_k = nn.Linear(latent_dim, latent_dim, bias=False)
        self.w_v = nn.Linear(latent_dim, latent_dim, bias=False)
        
        # Обучаемый параметр коэффициента шага ODE
        self.alpha_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, z_prev, s_prev):
        """
        Шаг перехода латентного состояния с численной Тейлор-стабилизацией.
        z_prev: предыдущее латентное состояние формы (B, D)
        s_prev: предыдущая матрица памяти EFLA формы (B, D, D)
        Возвращает:
            z_next: новое латентное состояние (B, D)
            s_next: обновленная матрица памяти (B, D, D)
        """
        k = self.w_k(z_prev)  # (B, D)
        v = self.w_v(z_prev)  # (B, D)
        
        # 1. Тейлор-стабилизированный гейт из модуля утилит
        alpha = taylor_stabilized_gate(k, self.alpha_scale)  # (B, 1)
        
        # 2. Нормализация Key
        k_norm = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-8)  # (B, D)
        
        # Вычисляем произведение k_norm^T * S_{t-1}
        k_s = torch.bmm(k_norm.unsqueeze(1), s_prev).squeeze(1)
        
        # Разность: v - k_norm^T * S_{t-1}
        diff = v - k_s  # (B, D)
        
        # Обновление матрицы памяти (ранг-1) с численной альфой: (B, D, 1) * (B, 1, D) -> (B, D, D)
        update = alpha.unsqueeze(2) * torch.bmm(k_norm.unsqueeze(2), diff.unsqueeze(1))
        s_next = s_prev + update
        
        # Вычисляем новое латентное состояние Z_t = S_next * k_norm
        z_next = torch.bmm(s_next, k_norm.unsqueeze(2)).squeeze(2)
        
        return z_next, s_next


def initialize_weights(module):
    """
    Применяет ортогональную инициализацию для 2D-матриц весов и Xavier/Glorot для остальных весов.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if module.weight.ndim == 2:
            nn.init.orthogonal_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
            
        if module.bias is not None:
            nn.init.zeros_(module.bias)
            
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class EFLAWorldModelTransition(nn.Module):
    """
    Каноническая модель латентного перехода, управляемая действиями, на базе EFLACell.
    """
    def __init__(self, cell, latent_dim=16, num_actions=4):
        super().__init__()
        self.cell = cell
        # Линейный слой для объединения латентного состояния и действия
        self.action_projection = nn.Linear(latent_dim + num_actions, latent_dim)

    def forward(self, z_prev, action_onehot, s_prev):
        # z_prev: (B, D), action_onehot: (B, num_actions)
        z_action = torch.cat([z_prev, action_onehot], dim=-1) # (B, D + num_actions)
        z_projected = torch.relu(self.action_projection(z_action)) # (B, D)
        z_next, s_next = self.cell(z_projected, s_prev)
        return z_next, s_next


class LatentDecoder(nn.Module):
    """
    Канонический декодер латентных представлений обратно в координаты.
    """
    def __init__(self, latent_dim=16, output_dim=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.fc(z)


class EFLAWorldModel(nn.Module):
    """
    Каноническая полная модель мира SEF-GRAM (VJEPA + EFLA + Transition + Decoder).
    Универсально настраивается под входную размерность, размерность скрытого слоя, выходную размерность и количество действий.
    """
    def __init__(self, input_dim=102, latent_dim=16, output_dim=2, num_actions=4):
        super().__init__()
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.transition = EFLAWorldModelTransition(self.cell, latent_dim=latent_dim, num_actions=num_actions)
        self.decoder = LatentDecoder(latent_dim=latent_dim, output_dim=output_dim)
        self.apply(initialize_weights)

    def forward_transition(self, s_t, action_onehot, s_prev):
        # s_t: (B, input_dim) - входное состояние
        mu_t, logvar_t = self.encoder(s_t)
        z_t = self.encoder.sample(mu_t, logvar_t)
        z_next, s_next = self.transition(z_t, action_onehot, s_prev)
        pred_coords = self.decoder(z_next)
        return pred_coords, z_next, s_next, mu_t, logvar_t
