import pytest
import torch
import numpy as np
from sef_gram.optimization import Muon
from sef_gram.blocks import get_lognormal_noise_intervals, DiffusionBlocksTransformer

def test_noise_intervals_correctness():
    """Проверяет правильность генерации интервалов шума"""
    num_blocks = 3
    intervals = get_lognormal_noise_intervals(num_blocks=num_blocks)
    
    assert len(intervals) == num_blocks + 1
    assert intervals[0] == 0.0
    assert intervals[-1] == 100.0
    
    # Интервалы должны строго возрастать
    for i in range(1, len(intervals)):
        assert intervals[i] > intervals[i-1]

def test_diffusion_blocks_gradient_isolation():
    """Проверяет, что градиенты рассчитываются только для активного блока слоев"""
    latent_dim = 16
    num_blocks = 3
    layers_per_block = 2
    
    model = DiffusionBlocksTransformer(
        latent_dim=latent_dim, 
        num_blocks=num_blocks, 
        layers_per_block=layers_per_block
    )
    
    # Инициализируем веса единичными матрицами, чтобы гарантировать прохождение градиентов
    for block in model.blocks:
        for layer in block:
            torch.nn.init.eye_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias)
    
    # Сделаем входы строго положительными, чтобы ReLU не занулял их
    x = torch.randn(2, latent_dim).abs() + 0.1
    
    # Активируем строго средний блок (индекс 1)
    model.set_active_block(active_block_idx=1)
    out = model(x, active_block_idx=1)
    loss = out.sum()
    loss.backward()
    
    # Блок 0: градиенты должны быть None, так как они отсечены на входе Блока 1
    for p in model.blocks[0].parameters():
        assert p.grad is None
        
    # Блок 2: градиенты должны быть None, так как этот блок заморожен с no_grad()
    for p in model.blocks[2].parameters():
        assert p.grad is None
        
    # Блок 1: градиенты должны быть успешно вычислены (не None)
    for p in model.blocks[1].parameters():
        assert p.grad is not None

def test_muon_optimizer_step():
    """Проверка работы шага оптимизатора Muon и ортогонализации"""
    # Создадим 2D веса и 1D смещение
    w = torch.nn.Parameter(torch.randn(8, 8))
    b = torch.nn.Parameter(torch.randn(8))
    
    optimizer = Muon([w, b], lr=0.1, momentum=0.9, ns_steps=5)
    
    # Зададим фиктивные градиенты
    w.grad = torch.randn(8, 8)
    b.grad = torch.randn(8)
    
    # Запомним старые значения весов
    w_old = w.clone()
    b_old = b.clone()
    
    # Делаем шаг оптимизации
    optimizer.step()
    
    # Параметры должны измениться
    assert not torch.allclose(w, w_old)
    assert not torch.allclose(b, b_old)
    
    # Проверим, что нет значений NaN
    assert not torch.isnan(w).any()
    assert not torch.isnan(b).any()
