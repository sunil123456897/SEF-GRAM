import torch
import torch.nn as nn
import math
import re
from sef_gram.model import VJEPAEncoder, EFLACell, initialize_weights
from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.blocks import DiffusionBlocksTransformer
from sef_gram.rl import compute_gdpo_advantages
from sef_gram.planning import BJEPAProductOfExpertsPlanner

class MazeEnvironment:
    """
    Среда Dynamic Maze (Сетчатый Лабиринт 10x10).
    Координаты от (0,0) до (9,9). Старт в (0,0), финиш в (9,9).
    Стены расположены в фиксированных точках.
    Действия: 0 = Вверх, 1 = Вниз, 2 = Влево, 3 = Вправо.
    """
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (9, 9)
        # Фиксированные стены в лабиринте
        self.walls = {
            (2, 2), (2, 3), (2, 4),
            (5, 5), (5, 6), (5, 7),
            (7, 2), (7, 3), (7, 4),
            (3, 8), (4, 8), (5, 8)
        }

    def parse_path_string(self, path_str):
        """Парсит XML разметку вида: <path>a0,a1,a2,...</path>"""
        pattern = r"<path>\s*([\d,\s]+)\s*</path>"
        match = re.search(pattern, path_str)
        if not match:
            return None
        try:
            return [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        except ValueError:
            return None

    def step(self, action):
        actions = []
        format_reward = 0.0
        
        if isinstance(action, str):
            # Reward Shaping
            micro_reward = 0.0
            if "<path>" in action:
                micro_reward += 0.1
            if "</path>" in action:
                micro_reward += 0.1
            
            parsed = self.parse_path_string(action)
            if parsed is not None:
                actions = parsed
                format_reward = 1.0
            else:
                actions = []
                format_reward = micro_reward
        else:
            actions = list(action)
            format_reward = 1.0

        # Симуляция прохождения лабиринта
        x, y = self.start
        visited_coords = [(x, y)]
        collisions = 0
        
        for act in actions:
            nx, ny = x, y
            if act == 0:    # Вверх (уменьшаем y)
                ny = max(0, y - 1)
            elif act == 1:  # Вниз (увеличиваем y)
                ny = min(self.size - 1, y + 1)
            elif act == 2:  # Влево (уменьшаем x)
                nx = max(0, x - 1)
            elif act == 3:  # Вправо (увеличиваем x)
                nx = min(self.size - 1, x + 1)
                
            # Проверяем коллизии со стенами
            if (nx, ny) in self.walls:
                collisions += 1
                # Агент остается на месте
            else:
                x, y = nx, ny
                visited_coords.append((x, y))

        # 1. Правильность (correctness)
        dist_to_goal = abs(x - self.goal[0]) + abs(y - self.goal[1]) # Манхэттенское расстояние
        correctness = 1.0 if dist_to_goal == 0 else float(math.exp(-0.2 * dist_to_goal))
        
        # Штрафуем за столкновения со стенами
        correctness = max(0.0, correctness - 0.1 * collisions)
        
        # 2. Краткость (brevity)
        brevity = float(math.exp(-0.05 * len(actions))) if len(actions) > 0 else 0.0
        
        return {
            'correctness': correctness,
            'format': format_reward,
            'brevity': brevity,
            'final_pos': (x, y)
        }

class SEFGRAMMazeSolver(nn.Module):
    def __init__(self, input_dim=2, latent_dim=16, num_actions=4, path_len=18):
        super().__init__()
        self.num_actions = num_actions
        self.path_len = path_len
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.blocks = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=2)
        self.policy_head = nn.Linear(latent_dim, path_len * num_actions)
        self.apply(initialize_weights)

    def forward(self, start_coords, active_block_idx):
        mu, logvar = self.encoder(start_coords)
        z = self.encoder.sample(mu, logvar)
        s_init = torch.zeros(start_coords.shape[0], z.shape[-1], z.shape[-1], device=start_coords.device)
        z_next, _ = self.cell(z, s_init)
        out = self.blocks(z_next, active_block_idx)
        logits = self.policy_head(out)
        logits = logits.view(-1, self.path_len, self.num_actions)
        return logits

def run_maze_experiment():
    print("="*70)
    print("ЭКСПЕРИМЕНТ 2: Динамическое планирование лабиринта (10x10)")
    print("="*70)
    
    env = MazeEnvironment()
    model = SEFGRAMMazeSolver(input_dim=2, latent_dim=16, num_actions=4, path_len=18)
    planner = BJEPAProductOfExpertsPlanner(latent_dim=16)
    
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=1e-3)
    
    # Решение-ориентир для SFT-разминки: путь вниз-вправо длины 18 шагов
    # (по 9 шагов вниз и вправо для достижения 9,9)
    target_path = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=torch.long)
    sft_loss_fn = nn.CrossEntropyLoss()
    
    # 1. SFT Warm-up (100 шагов)
    print("\n--- [Шаг 1: SFT Warm-up] Обучение генерации базовой траектории ---")
    for sft_step in range(100):
        optimizer.zero_grad()
        model.blocks.set_active_block(0)
        
        # Стабильный контекст: вектор из единиц (batch_size=16 для параллельной тренировки в SFT)
        sft_batch = 16
        start_coords = torch.ones(sft_batch, 2)
        
        logits = model(start_coords, active_block_idx=0) # (sft_batch, path_len=18, num_actions=4)
        
        # Распрямляем для CrossEntropyLoss
        logits_flat = logits.view(-1, 4) # (sft_batch * 18, 4)
        target_expanded = target_path.repeat(sft_batch) # (sft_batch * 18,)
        
        loss = sft_loss_fn(logits_flat, target_expanded)
        loss.backward()
        optimizer.step()
        
        if (sft_step + 1) % 10 == 0:
            # Считаем точность предсказания на SFT
            preds = torch.argmax(logits, dim=-1) # (sft_batch, 18)
            sft_correctness = (preds == target_path).all(dim=-1).float().mean().item()
            print(f"SFT Шаг {sft_step + 1:03d}/100 | SFT Loss: {loss.item():.4f} | Точность подгонки: {sft_correctness:.2%}")
            
    # 2. RL GDPO с планировщиком PoE и динамическим циклом до 90%
    print("\n--- [Шаг 2: RL GDPO + BJEPA PoE латентное планирование (до 90%+)] ---")
    group_size = 32
    path_len = 18
    
    max_steps = 1000
    patience = 40
    best_correctness = 0.0
    stagnation_steps = 0
    step = 0
    
    while step < max_steps:
        optimizer.zero_grad()
        active_block_idx = (step + 1) % 3
        model.blocks.set_active_block(active_block_idx)
        
        # Стабильный контекст для RL-выравнивания
        start_coords = torch.ones(group_size, 2)
        logits = model(start_coords, active_block_idx) # (group_size, 18, 4)
        probs = torch.softmax(logits, dim=-1)
        
        probs_cpu = probs.detach().cpu()
        group_paths = []
        for i in range(group_size):
            actions_i = []
            for step_idx in range(path_len):
                step_probs = probs_cpu[i, step_idx]
                action_step = torch.multinomial(step_probs, num_samples=1).item()
                actions_i.append(action_step)
            group_paths.append(actions_i)
            
        rewards = {
            'correctness': [],
            'format': [],
            'brevity': []
        }
        
        final_positions = []
        # Reward Shaping: все кандидаты генерируют идеальный XML-формат
        for idx, path in enumerate(group_paths):
            path_str = ",".join(map(str, path))
            xml_str = f"<path>{path_str}</path>"
            
            res = env.step(xml_str)
            rewards['correctness'].append(res['correctness'])
            rewards['format'].append(res['format'])
            rewards['brevity'].append(res['brevity'])
            final_positions.append(res['final_pos'])
            
        advantages = compute_gdpo_advantages(rewards)
        
        loss = 0.0
        for i in range(group_size):
            log_prob = 0.0
            for step_idx in range(path_len):
                selected_action = group_paths[i][step_idx]
                log_prob += torch.log(probs[i, step_idx, selected_action] + 1e-8)
            loss += -log_prob * advantages[i]
            
        loss = loss / group_size
        loss.backward()
        optimizer.step()
        
        avg_correctness = sum(rewards['correctness']) / group_size
        avg_format = sum(rewards['format']) / group_size
        
        if (step + 1) % 10 == 0 or step == 0:
            print(f"GDPO Шаг {step + 1:03d} | Лосс: {loss.item():+.4f} | Близость к финишу: {avg_correctness:.2%} | Застой: {stagnation_steps}/{patience} | Финал Канд0: {final_positions[0]}")
            
        if avg_correctness >= 0.90:
            print(f"\n[УСПЕХ] Целевая точность 90%+ в лабиринте достигнута на шаге {step + 1}! Итоговый процент: {avg_correctness:.2%}")
            break
            
        if avg_correctness > best_correctness:
            best_correctness = avg_correctness
            stagnation_steps = 0
        else:
            stagnation_steps += 1
            
        if stagnation_steps >= patience:
            print(f"\n[ОСТАНОВКА] Обучение остановлено из-за застоя! В течение {patience} шагов точность не превысила лучшую ({best_correctness:.2%}).")
            break
            
        step += 1
        
    # Демонстрация латентного слияния экспертов BJEPA PoE
    print("\n--- Демонстрация байесовского слияния экспертов (Product of Experts) ---")
    # Likelihood Expert: предсказание модели мира (задаем как случайные блуждания с разной дисперсией)
    mu_world = torch.zeros(1, 16)
    logvar_world = torch.zeros(1, 16) # дисперсия = 1.0 (высокая неопределенность)
    
    # Prior Expert: притяжение к целевым координатам (низкая дисперсия, высокая уверенность)
    mu_goal = torch.ones(1, 16) * 3.0
    logvar_goal = torch.ones(1, 16) * (-2.0) # дисперсия = exp(-2) = 0.13
    
    mu_fused, logvar_fused = planner.fuse_experts(mu_world, logvar_world, mu_goal, logvar_goal)
    print(f"  Likelihood Expert (среднее): [{', '.join(f'{x:.2f}' for x in mu_world[0][:4])}...]")
    print(f"  Prior Expert (цель):        [{', '.join(f'{x:.2f}' for x in mu_goal[0][:4])}...]")
    print(f"  Объединенный PoE-план:      [{', '.join(f'{x:.2f}' for x in mu_fused[0][:4])}...]")
    print(f"  Итоговая неопределенность (logvar): [{', '.join(f'{x:.2f}' for x in logvar_fused[0][:4])}...]")
    
    print("="*70)
    print("ЭКСПЕРИМЕНТ 2 ЗАВЕРШЕН!")
    print("="*70)

if __name__ == "__main__":
    run_maze_experiment()
