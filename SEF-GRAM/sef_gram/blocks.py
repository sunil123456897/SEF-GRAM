import torch
import numpy as np
import random
from scipy.stats import lognorm

def get_lognormal_noise_intervals(num_blocks=3, loc=0.0, scale=1.0):
    """
    Разбивает интегральную функцию распределения (CDF) лог-нормального шума
    на num_blocks равновероятностных интервалов.
    Возвращает границы интервалов в виде массива.
    """
    probabilities = np.linspace(0.0, 1.0, num_blocks + 1)
    quantiles = lognorm.ppf(probabilities, s=scale, scale=np.exp(loc))
    quantiles[0] = 0.0
    quantiles[-1] = 100.0
    return quantiles


class DiffusionBlocksTransformer(torch.nn.Module):
    """
    Трансформер с поблочным проходом градиентов (DiffusionBlocks).
    Разделяет слои на B блоков.
    Реализует Stochastic Block Scheduling и EMA Coordinate Anchoring для
    полной стабилизации геометрии скрытых векторов и исключения координатного дрейфа.
    """
    def __init__(self, latent_dim=512, num_blocks=3, layers_per_block=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        
        # Список блоков слоев
        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Linear(latent_dim, latent_dim)
                for _ in range(layers_per_block)
            ]) for _ in range(num_blocks)
        ])
        
        # Общий LayerNorm
        self.ln = torch.nn.LayerNorm(latent_dim)
        
        # EMA Coordinate Anchoring буферы
        self.ema_gamma = 0.99
        self.register_buffer('ema_initialized', torch.tensor(False))
        self.ema_buffers = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)
            for _ in range(num_blocks)
        ])

    def set_active_block(self, active_block_idx):
        """
        Включает requires_grad только для параметров активного блока.
        Поддерживает стохастический выбор при active_block_idx < 0.
        """
        if active_block_idx < 0:
            active_block_idx = random.randint(0, self.num_blocks - 1)
            
        self.current_active_block = active_block_idx
        for b_idx, block in enumerate(self.blocks):
            if b_idx == active_block_idx:
                for p in block.parameters():
                    p.requires_grad = True
            else:
                for p in block.parameters():
                    p.requires_grad = False

    def forward(self, x, active_block_idx):
        """
        x: входные скрытые вектора (B, latent_dim)
        active_block_idx: индекс блока, для которого будет рассчитан граф градиентов.
                          Если active_block_idx < 0, то используется стохастический выбор.
        """
        if active_block_idx < 0:
            if not hasattr(self, 'current_active_block'):
                self.set_active_block(-1)
            active_block_idx = self.current_active_block
            
        out = x
        for b_idx, block in enumerate(self.blocks):
            if b_idx < active_block_idx:
                # Upstream-блоки: вычисления без градиентов для экономии памяти
                with torch.no_grad():
                    for layer in block:
                        out = torch.nn.functional.relu(layer(out))
                
                # EMA Coordinate Anchoring для фиксации скрытой геометрии
                if self.training:
                    mean_coords = out.mean(dim=0, keepdim=True).detach()
                    if not self.ema_initialized:
                        self.ema_buffers[b_idx].copy_(mean_coords)
                    else:
                        self.ema_buffers[b_idx].copy_(self.ema_gamma * self.ema_buffers[b_idx] + (1.0 - self.ema_gamma) * mean_coords)
                    
                    # Мягко притягиваем скрытые координаты к EMA для подавления дрейфа
                    out = out + 0.01 * (self.ema_buffers[b_idx] - out)
                    
                out = out.detach()
            else:
                # Active и Downstream-блоки
                for layer in block:
                    out = torch.nn.functional.relu(layer(out))
                    
        if self.training and not self.ema_initialized:
            self.ema_initialized.copy_(torch.tensor(True))
            
        return self.ln(out)
