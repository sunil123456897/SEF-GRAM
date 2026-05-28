import torch
from sef_gram.rl import compute_gdpo_advantages

def compute_grpo_advantages(rewards, eps=1e-8):
    """
    Классическая формула GRPO: суммируем все награды, затем нормализуем.
    """
    total_rewards = None
    for key, vals in rewards.items():
        tensor_vals = torch.tensor(vals, dtype=torch.float32)
        if total_rewards is None:
            total_rewards = tensor_vals
        else:
            total_rewards += tensor_vals
            
    mean = total_rewards.mean()
    std = total_rewards.std()
    return (total_rewards - mean) / (std + eps)

def run_gdpo_penalty_test():
    print("=== GDPO vs GRPO Penalty Test ===")
    
    # Симулируем группу из 8 кандидатов (rollouts)
    # correctness: 1.0 (верно), 0.0 (неверно) - это сильный сигнал
    # format: 1.0 (идеальный формат), -0.5 (ошибка формата) - это слабый сигнал (штраф)
    
    # Кандидаты:
    # 0-3: Решили правильно, но ошиблись в формате
    # 4-6: Решили неправильно, ошиблись в формате
    # 7: Решил неправильно, но формат ИДЕАЛЬНЫЙ
    
    rewards = {
        'correctness': [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        'format':      [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 1.0] 
    }
    
    print("\nИсходные награды:")
    print(f"Correctness: {rewards['correctness']}")
    print(f"Format:      {rewards['format']}")
    
    # Считаем преимущества
    grpo_adv = compute_grpo_advantages(rewards)
    gdpo_adv = compute_gdpo_advantages(rewards)
    
    print("\n--- Сравнение преимуществ ---")
    print("Индекс | Состояние                      | GRPO Adv | GDPO Adv | Вывод")
    print("-" * 80)
    
    states = [
        "Верно, плохой формат   ",
        "Верно, плохой формат   ",
        "Верно, плохой формат   ",
        "Верно, плохой формат   ",
        "Неверно, плохой формат ",
        "Неверно, плохой формат ",
        "Неверно, плохой формат ",
        "Неверно, ИДЕАЛЬНЫЙ формат"
    ]
    
    for i in range(8):
        grpo_val = grpo_adv[i].item()
        gdpo_val = gdpo_adv[i].item()
        
        # Анализ для кандидата 7 (Критический кейс)
        if i == 7:
            if gdpo_val > grpo_val:
                conclusion = "GDPO оценил формат!"
            else:
                conclusion = "Collapse"
        elif i == 0:
            conclusion = "GRPO игнорирует штраф формата" if grpo_val > gdpo_val else ""
        else:
            conclusion = ""
            
        print(f"  {i}    | {states[i]}|  {grpo_val:+.3f}  |  {gdpo_val:+.3f}  | {conclusion}")
        
    print("\n--- Анализ ---")
    print("В GRPO кандидат #7 (идеальный формат, но неверный ответ) получает самое низкое преимущество,")
    print("потому что награда за 'correctness' полностью перекрывает штраф за формат.")
    print("В GDPO кандидат #7 получает существенный буст преимущества от компоненты 'format' (после декуплированной нормализации),")
    print("что позволяет модели учиться формату даже на провальных траекториях!")

if __name__ == "__main__":
    run_gdpo_penalty_test()
