import pytest
import torch
from sef_gram.model import VJEPAEncoder, EFLACell

def test_vjepa_encoder_shapes():
    """Проверка выходных размерностей кодировщика VJEPA"""
    input_dim = 32
    latent_dim = 256
    batch_size = 8
    
    encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
    x = torch.randn(batch_size, input_dim)
    
    mu, logvar = encoder(x)
    
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)

def test_vjepa_reparameterization_flow():
    """Проверка прохождения градиентов через стохастическое сэмплирование"""
    encoder = VJEPAEncoder(input_dim=16, latent_dim=64)
    x = torch.randn(4, 16, requires_grad=True)
    
    mu, logvar = encoder(x)
    z = encoder.sample(mu, logvar)
    
    # Вычисляем лосс и запускаем бэкпроп
    loss = z.pow(2).sum()
    loss.backward()
    
    # Градиенты должны корректно течь ко входу x
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert x.grad.abs().sum() > 0.0

def test_efla_cell_shapes_and_stability():
    """Проверка размерностей и стабильности шагов EFLA ODE"""
    latent_dim = 128
    batch_size = 4
    
    cell = EFLACell(latent_dim=latent_dim)
    
    z_prev = torch.randn(batch_size, latent_dim)
    s_prev = torch.zeros(batch_size, latent_dim, latent_dim) # нулевая инициализация памяти
    
    z_next, s_next = cell(z_prev, s_prev)
    
    assert z_next.shape == (batch_size, latent_dim)
    assert s_next.shape == (batch_size, latent_dim, latent_dim)
    assert not torch.isnan(z_next).any()
    assert not torch.isnan(s_next).any()

def test_efla_analytical_precision():
    """Проверка точной интеграции: EFLA не должна приводить к сингулярности при последовательных шагах"""
    latent_dim = 32
    batch_size = 2
    cell = EFLACell(latent_dim=latent_dim)
    
    z = torch.randn(batch_size, latent_dim)
    s = torch.zeros(batch_size, latent_dim, latent_dim)
    
    # Запустим длинную траекторию из 100 шагов
    for _ in range(100):
        z, s = cell(z, s)
        
    assert not torch.isnan(z).any()
    assert not torch.isnan(s).any()
    # Матрица памяти не должна улетать в бесконечность
    assert s.abs().max().item() < 1e5
