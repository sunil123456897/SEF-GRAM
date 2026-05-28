import pytest
import torch
import math
from sef_gram.rl import compute_gdpo_advantages
from sef_gram.planning import BJEPAProductOfExpertsPlanner
from sef_gram.environment import NQueensEnvironment

def test_gdpo_advantages_normalization():
    """Проверка расчета и групповой z-score нормализации в GDPO"""
    rewards = {
        'correctness': [1.0, 0.0, 1.0, 0.0],
        'format': [1.0, 1.0, 0.0, 0.0],
        'brevity': [0.9, 0.5, 0.9, 0.2]
    }
    
    advs = compute_gdpo_advantages(rewards)
    assert advs.shape == (4,)
    
    # Поскольку преимущества нормализуются с нулевым средним по каждой группе,
    # сумма всех преимуществ в батче должна быть близка к нулю.
    assert abs(advs.sum().item()) < 1e-5

def test_poe_gaussian_fusion():
    """Проверяет слияние распределений двух экспертов (Product of Experts)"""
    planner = BJEPAProductOfExpertsPlanner(latent_dim=4)
    
    # Эксперт 1: mu=0, logvar=0 (variance=1)
    mu1 = torch.zeros(2, 4)
    logvar1 = torch.zeros(2, 4)
    
    # Эксперт 2: mu=2, logvar=0 (variance=1)
    mu2 = torch.ones(2, 4) * 2.0
    logvar2 = torch.zeros(2, 4)
    
    # Результат слияния двух одинаково уверенных распределений должен дать mu=1.0 и меньшую дисперсию
    mu_f, logvar_f = planner.fuse_experts(mu1, logvar1, mu2, logvar2)
    
    assert torch.allclose(mu_f, torch.ones(2, 4) * 1.0)
    # Дисперсия слияния: 1/var = 1/1 + 1/1 = 2 -> var = 0.5 -> logvar = -0.693
    expected_logvar = -math.log(2.0)
    assert torch.allclose(logvar_f, torch.ones(2, 4) * expected_logvar, atol=1e-5)

def test_nqueens_environment_validation():
    """Проверка расчета наград и парсинга XML в среде N-Queens"""
    env = NQueensEnvironment(n=4)
    
    # 1. Валидный ответ списком (симметричная расстановка ферзей на доске 4x4)
    res_list = env.step([2, 0, 3, 1])
    assert res_list['correctness'] == 1.0
    assert res_list['format'] == 1.0
    assert res_list['brevity'] == float(math.exp(-0.4))
    
    # 2. Невалидный ответ списком (конфликтующие ферзи)
    res_conflict = env.step([0, 0, 0, 0])
    assert res_conflict['correctness'] == 0.0
    
    # 3. Валидный ответ строкой с XML разметкой
    res_xml_valid = env.step("<answer>2, 0, 3, 1</answer>")
    assert res_xml_valid['correctness'] == 1.0
    assert res_xml_valid['format'] == 1.0
    
    # 4. Невалидный XML формат
    res_xml_invalid = env.step("2, 0, 3, 1")
    assert res_xml_invalid['correctness'] == 0.0
    assert res_xml_invalid['format'] == 0.0
