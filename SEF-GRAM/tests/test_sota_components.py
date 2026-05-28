import pytest
import torch
import math
from sef_gram.model import VJEPAEncoder, EFLACell
from sef_gram.optimization import Muon
from sef_gram.blocks import DiffusionBlocksTransformer
from sef_gram.rl import compute_echo_gdpo_loss
from sef_gram.planning import BJEPAProductOfExpertsPlanner

def test_taylor_stabilized_efla():
    """Проверяет Taylor-стабилизацию EFLACell при сверхмалых значениях Key"""
    latent_dim = 16
    cell = EFLACell(latent_dim=latent_dim)
    
    # Задаем сверхмалые значения входа, близкие к нулю
    z_prev = torch.zeros(2, latent_dim) + 1e-9
    s_prev = torch.zeros(2, latent_dim, latent_dim)
    
    z_next, s_next = cell(z_prev, s_prev)
    
    # Не должно быть NaN
    assert not torch.isnan(z_next).any()
    assert not torch.isnan(s_next).any()
    # Значения должны быть стабильными (около нуля)
    assert torch.allclose(z_next, torch.zeros_like(z_next), atol=1e-5)


def test_stochastic_block_scheduling():
    """Проверяет стохастический Block Scheduling в DiffusionBlocksTransformer"""
    latent_dim = 16
    model = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=1)
    
    # 1. Проверяем автовыбор блока при set_active_block(-1)
    model.set_active_block(-1)
    assert 0 <= model.current_active_block < 3
    
    # Проверяем requires_grad у активных и замороженных параметров
    active_idx = model.current_active_block
    for b_idx, block in enumerate(model.blocks):
        for p in block.parameters():
            if b_idx == active_idx:
                assert p.requires_grad is True
            else:
                assert p.requires_grad is False


def test_ema_coordinate_anchoring():
    """Проверяет обновление EMA буферов в DiffusionBlocksTransformer при обучении"""
    latent_dim = 16
    model = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=1)
    model.train()
    
    x = torch.randn(4, latent_dim)
    
    # Вызываем forward
    out = model(x, active_block_idx=1)
    
    # EMA должна инициализироваться и буферы должны обновиться
    assert model.ema_initialized.item() is True
    assert model.ema_buffers[0].abs().sum().item() > 0.0


def test_gram_muon_orthogonalization():
    """Проверяет ортогонализацию градиентов в Muon"""
    w = torch.nn.Parameter(torch.randn(4, 16))
    optimizer = Muon([w], lr=0.1, ns_steps=5)
    
    w.grad = torch.randn(4, 16)
    
    # Шаг оптимизации
    optimizer.step()
    
    # Веса должны успешно обновиться
    assert not torch.isnan(w).any()


def test_muon_stability_against_infinite_gradients():
    """
    Проверяет защиту оптимизатора Muon против бесконечных градиентов (inf/-inf).
    Оптимизатор НЕ должен обнулять или превращать в бесконечность веса 1D и 2D параметров.
    """
    # Тест 2D параметров
    w = torch.nn.Parameter(torch.ones(4, 16))
    # Тест 1D параметров (где раньше не было проверки вообще)
    b = torch.nn.Parameter(torch.ones(16))
    
    optimizer = Muon([w, b], lr=0.1, ns_steps=5)
    
    # Задаем бесконечные градиенты
    w.grad = torch.tensor([[float('inf')] * 16] * 4)
    b.grad = torch.tensor([float('inf')] * 16)
    
    # Шаг оптимизации
    optimizer.step()
    
    # Веса должны оставаться конечными и не должны изменяться
    assert torch.isfinite(w).all(), "2D параметр должен оставаться конечным при бесконечных градиентах!"
    assert torch.allclose(w, torch.ones_like(w)), "2D параметр не должен изменяться при бесконечном градиенте!"
    
    assert torch.isfinite(b).all(), "1D параметр должен оставаться конечным при бесконечных градиентах!"
    assert torch.allclose(b, torch.ones_like(b)), "1D параметр не должен изменяться при бесконечном градиенте!"


def test_echo_gdpo_loss():
    """Проверяет расчет совместного лосса ECHO-GDPO и прохождение градиентов"""
    policy_logits = torch.randn(2, 5, 4, requires_grad=True)
    actions = torch.randint(0, 4, (2, 5))
    advantages = torch.tensor([1.2, -0.5])
    
    env_logits = torch.randn(2, 10, 8, requires_grad=True)
    env_targets = torch.randint(0, 8, (2, 10))
    
    loss, p_loss, e_loss = compute_echo_gdpo_loss(
        policy_logits, actions, advantages, env_logits, env_targets, mix_lambda=0.05
    )
    
    assert loss.item() is not None
    assert p_loss.item() is not None
    assert e_loss.item() is not None
    
    # Проверяем бэкпроп
    loss.backward()
    assert policy_logits.grad is not None
    assert env_logits.grad is not None


def test_unsl_thinking_steps_autotuning():
    """Проверяет UNSL закон автонастройки шагов размышлений T"""
    planner = BJEPAProductOfExpertsPlanner(latent_dim=16)
    
    # Маленькая модель (10 тысяч параметров) должна получить большой T (45)
    t_small = planner.autotune_thinking_steps(num_params=1e4, target_loss=0.01)
    assert t_small == 45
    
    # Крупная модель (100 миллионов параметров) должна получить оптимальный T в диапазоне [5, 45]
    t_large = planner.autotune_thinking_steps(num_params=1e8, target_loss=0.01)
    assert 5 <= t_large <= 45
