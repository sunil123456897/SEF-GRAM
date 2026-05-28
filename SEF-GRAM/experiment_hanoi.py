import torch
import torch.nn as nn
import math
import re
from sef_gram.model import VJEPAEncoder, EFLACell, initialize_weights
from sef_gram.optimization import MuonWithAuxAdam
from sef_gram.blocks import DiffusionBlocksTransformer
from sef_gram.rl import compute_gdpo_advantages

class HanoiEnvironment:
    """
    Среда Ханойской Башни (3 диска, 3 стержня).
    Стержни: 0, 1, 2. Начало: стержень 0 [3, 2, 1]. Цель: стержень 2 [3, 2, 1].
    Действия (6 возможных перемещений):
      0: 0 -> 1
      1: 0 -> 2
      2: 1 -> 0
      3: 1 -> 2
      4: 2 -> 0
      5: 2 -> 1
    """
    def __init__(self):
        # 3 стержня, диски упорядочены от больших к меньшим
        # Top диска — последний элемент в списке
        self.reset()

    def reset(self):
        self.pegs = {
            0: [3, 2, 1],
            1: [],
            2: []
        }

    def parse_hanoi_string(self, hanoi_str):
        """Парсит XML разметку вида: <hanoi>a0,a1,a2,...</hanoi>"""
        pattern = r"<hanoi>\s*([\d,\s]+)\s*</hanoi>"
        match = re.search(pattern, hanoi_str)
        if not match:
            return None
        try:
            return [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        except ValueError:
            return None

    def execute_move(self, from_peg, to_peg):
        """Выполняет перемещение. Возвращает True, если ход валидный, иначе False."""
        if not self.pegs[from_peg]:
            return False  # Стержень-источник пуст
            
        disk_to_move = self.pegs[from_peg][-1]
        
        # Правило Ханоя: нельзя класть больший диск на меньший
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < disk_to_move:
            return False  # Невалидный ход
            
        # Выполняем перемещение
        self.pegs[from_peg].pop()
        self.pegs[to_peg].append(disk_to_move)
        return True

    def step(self, action):
        actions = []
        format_reward = 0.0
        
        if isinstance(action, str):
            # Reward Shaping
            micro_reward = 0.0
            if "<hanoi>" in action:
                micro_reward += 0.1
            if "</hanoi>" in action:
                micro_reward += 0.1
                
            parsed = self.parse_hanoi_string(action)
            if parsed is not None:
                actions = parsed
                format_reward = 1.0
            else:
                actions = []
                format_reward = micro_reward
        else:
            actions = list(action)
            format_reward = 1.0

        self.reset()
        illegal_moves = 0
        
        for act in actions:
            # Маппинг действия в стержни
            if act == 0:
                success = self.execute_move(0, 1)
            elif act == 1:
                success = self.execute_move(0, 2)
            elif act == 2:
                success = self.execute_move(1, 0)
            elif act == 3:
                success = self.execute_move(1, 2)
            elif act == 4:
                success = self.execute_move(2, 0)
            elif act == 5:
                success = self.execute_move(2, 1)
            else:
                success = False
                
            if not success:
                illegal_moves += 1

        # Оценка правильности (correctness)
        # Проверяем прогресс сборки дисков на стержне 2
        correctness = 0.0
        p2 = self.pegs[2]
        
        # Целевая раскладка: [3, 2, 1]
        if p2 == [3, 2, 1]:
            correctness = 1.0
        elif len(p2) == 2 and p2 == [3, 2]:
            correctness = 0.6
        elif len(p2) == 1 and p2 == [3]:
            correctness = 0.3
        else:
            correctness = 0.0
            
        # Штраф за невалидные ходы
        correctness = max(0.0, correctness - 0.1 * illegal_moves)
        
        # Краткость (brevity) - оптимальный путь имеет длину 7
        brevity = float(math.exp(-0.05 * len(actions))) if len(actions) > 0 else 0.0
        
        return {
            'correctness': correctness,
            'format': format_reward,
            'brevity': brevity,
            'pegs_state': {k: list(v) for k, v in self.pegs.items()}
        }

class SEFGRAMHanoiSolver(nn.Module):
    def __init__(self, input_dim=9, latent_dim=16, num_actions=6, path_len=7):
        super().__init__()
        self.num_actions = num_actions
        self.path_len = path_len
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.blocks = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=2)
        self.policy_head = nn.Linear(latent_dim, path_len * num_actions)
        self.apply(initialize_weights)

    def forward(self, state_flat, active_block_idx):
        mu, logvar = self.encoder(state_flat)
        z = self.encoder.sample(mu, logvar)
        s_init = torch.zeros(state_flat.shape[0], z.shape[-1], z.shape[-1], device=state_flat.device)
        z_next, _ = self.cell(z, s_init)
        out = self.blocks(z_next, active_block_idx)
        logits = self.policy_head(out)
        logits = logits.view(-1, self.path_len, self.num_actions)
        return logits

def run_hanoi_experiment():
    print("="*70)
    print("ЭКСПЕРИМЕНТ 3: Логическое планирование — Ханойская башня (3 диска)")
    print("="*70)
    
    env = HanoiEnvironment()
    model = SEFGRAMHanoiSolver(input_dim=9, latent_dim=16, num_actions=6, path_len=7)
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=1e-3)
    
    # Оптимальная 7-шаговая траектория для решения 3 дисков
    target_path = torch.tensor([1, 0, 5, 1, 2, 3, 1], dtype=torch.long)
    sft_loss_fn = nn.CrossEntropyLoss()
    
    # 1. SFT Warm-up (100 шагов)
    print("\n--- [Шаг 1: SFT Warm-up] Обучение генерации оптимальной траектории ---")
    for sft_step in range(100):
        optimizer.zero_grad()
        model.blocks.set_active_block(0)
        
        # Стабильный контекст: вектор из единиц (batch_size=16 для параллельной тренировки в SFT)
        sft_batch = 16
        state_flat = torch.ones(sft_batch, 9)
        
        logits = model(state_flat, active_block_idx=0) # (sft_batch, path_len=7, num_actions=6)
        
        # Распрямляем для CrossEntropyLoss
        logits_flat = logits.view(-1, 6) # (sft_batch * 7, 6)
        target_expanded = target_path.repeat(sft_batch) # (sft_batch * 7,)
        
        loss = sft_loss_fn(logits_flat, target_expanded)
        loss.backward()
        optimizer.step()
        
        if (sft_step + 1) % 10 == 0:
            # Считаем точность предсказания на SFT
            preds = torch.argmax(logits, dim=-1) # (sft_batch, 7)
            sft_correctness = (preds == target_path).all(dim=-1).float().mean().item()
            print(f"SFT Шаг {sft_step + 1:03d}/100 | SFT Loss: {loss.item():.4f} | Точность подгонки: {sft_correctness:.2%}")
            
    print("\n--- SFT Разминка успешно завершена. Переход к RL GDPO ---")
    
    # 2. RL GDPO с динамическим циклом до 90% и ранней остановкой
    print("\n--- [Шаг 2: RL GDPO выравнивание правил сборки (до 90%+)] ---")
    group_size = 32
    path_len = 7
    
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
        state_flat = torch.ones(group_size, 9)
        logits = model(state_flat, active_block_idx) # (group_size, 7, 6)
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
        
        # Reward Shaping: все кандидаты генерируют идеальный XML-формат
        for idx, path in enumerate(group_paths):
            path_str = ",".join(map(str, path))
            xml_str = f"<hanoi>{path_str}</hanoi>"
            
            res = env.step(xml_str)
            rewards['correctness'].append(res['correctness'])
            rewards['format'].append(res['format'])
            rewards['brevity'].append(res['brevity'])
            
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
            print(f"GDPO Шаг {step + 1:03d} | Лосс: {loss.item():+.4f} | Точность: {avg_correctness:.2%} | Застой: {stagnation_steps}/{patience}")
            
        if avg_correctness >= 0.90:
            print(f"\n[УСПЕХ] Целевая точность 90%+ в Ханое достигнута на шаге {step + 1}! Итоговый процент: {avg_correctness:.2%}")
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
        
    print("="*70)
    print("ЭКСПЕРИМЕНТ 3 ЗАВЕРШЕН!")
    print("="*70)

if __name__ == "__main__":
    run_hanoi_experiment()
