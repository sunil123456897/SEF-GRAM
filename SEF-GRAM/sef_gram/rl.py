import torch
import torch.nn as nn

def compute_gdpo_advantages(rewards, eps=1e-8):
    """
    Вычисляет декуплированные преимущества (decoupled advantages) для алгоритма GDPO.
    Производит групповую нормализацию преимуществ раздельно для каждой цели,
    после чего суммирует их.
    rewards: словарь списков наград по целям, например:
             {
                 'correctness': [r_1, r_2, ..., r_N],
                 'format': [f_1, f_2, ..., f_N],
                 'brevity': [b_1, b_2, ..., b_N]
             }
    N - размер группы кандидатов (group size).
    Возвращает:
        advantages: тензор преимуществ формы (N,)
    """
    advantages = None
    
    for key, vals in rewards.items():
        tensor_vals = torch.tensor(vals, dtype=torch.float32)
        if len(vals) > 1:
            mean = tensor_vals.mean()
            std = tensor_vals.std()
            # Групповая нормализация (z-score) раздельно для каждой награды
            normed = (tensor_vals - mean) / (std + eps)
        else:
            normed = torch.zeros_like(tensor_vals)
            
        if advantages is None:
            advantages = normed
        else:
            advantages += normed
            
    return advantages


def compute_echo_gdpo_loss(policy_logits, actions, advantages, env_logits, env_targets, mix_lambda=0.05):
    """
    Вычисляет совместный гибридный лосс ECHO-GDPO.
    policy_logits: логиты предсказания действий формы (B, path_len, num_actions)
    actions: выбранные действия формы (B, path_len)
    advantages: нормализованные преимущества GDPO формы (B,)
    env_logits: логиты предсказания ответов среды формы (B, target_len, env_dim)
    env_targets: реальные ответы среды формы (B, target_len)
    mix_lambda: коэффициент смешивания для лосса среды
    """
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    
    # 1. Лосс политики (GDPO)
    # Вычисляем negative log likelihood для выбранных действий
    B, path_len, num_actions = policy_logits.shape
    policy_logits_flat = policy_logits.view(-1, num_actions)
    actions_flat = actions.view(-1)
    
    nll = loss_fn(policy_logits_flat, actions_flat)  # (B * path_len,)
    nll = nll.view(B, path_len).sum(dim=-1)  # Суммируем по шагам траектории -> (B,)
    
    # Преимущества GDPO умножаются на nll
    policy_loss = torch.mean(nll * advantages)
    
    # 2. Лосс модели мира (ECHO Env Loss)
    # Кросс-энтропия предсказания ответа среды
    B_env, target_len, env_dim = env_logits.shape
    env_logits_flat = env_logits.view(-1, env_dim)
    env_targets_flat = env_targets.view(-1)
    
    env_ce = loss_fn(env_logits_flat, env_targets_flat)  # (B * target_len,)
    env_loss = torch.mean(env_ce)
    
    # Совместный гибридный лосс ECHO-GDPO
    total_loss = policy_loss + mix_lambda * env_loss
    
    return total_loss, policy_loss, env_loss
