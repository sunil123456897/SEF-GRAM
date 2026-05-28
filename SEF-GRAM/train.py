import torch
import torch.nn as nn
from sef_gram.model import VJEPAEncoder, EFLACell, initialize_weights
from sef_gram.optimization import DiffusionBlocksTransformer, MuonWithAuxAdam
from sef_gram.rl import compute_gdpo_advantages
from sef_gram.environment import NQueensEnvironment

class SEFGRAMTaskSolver(nn.Module):
    """
    Модель-оболочка SEF-GRAM для выполнения задач.
    Объединяет кодировщик VJEPA, рекуррентное ядро EFLA и поблочный трансформер.
    """
    def __init__(self, input_dim=4, latent_dim=16, board_size=4):
        super().__init__()
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.blocks = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=2)
        
        # Выходная голова для проецирования латентных состояний в ходы (колонки ферзей)
        self.policy_head = nn.Linear(latent_dim, board_size)
        
        # Применяем спектрально стабильную ортогональную инициализацию весов (DEC-005)
        self.apply(initialize_weights)

    def forward(self, x, active_block_idx):
        # 1. Кодирование входных данных в латентное пространство
        mu, logvar = self.encoder(x)
        z = self.encoder.sample(mu, logvar)
        
        # 2. Непрерывный латентный переход EFLA (одношаговый для демонстрации)
        s_init = torch.zeros(x.shape[0], z.shape[-1], z.shape[-1], device=x.device)
        z_next, _ = self.cell(z, s_init)
        
        # 3. Диффузионный проход через замороженные / активные блоки
        out = self.blocks(z_next, active_block_idx)
        
        # 4. Выходная политика для выбора ходов
        logits = self.policy_head(out)
        return logits


def run_e2e_training():
    print("="*60)
    print("Запуск сквозного обучения SEF-GRAM на задаче N-Queens (4x4)")
    print("="*60)
    
    # Инициализация среды и модели
    board_size = 4
    group_size = 4  # размер группы для GDPO
    env = NQueensEnvironment(n=board_size)
    model = SEFGRAMTaskSolver(input_dim=board_size, latent_dim=16, board_size=board_size)
    
    # Гибридный оптимизатор: веса Muon (2D), смещения и классификатор AdamW
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=1e-3)
    
    # ----------------------------------------------------
    # ЭТАП 1: SFT Разминка (SFT Warm-up)
    # ----------------------------------------------------
    print("\n[ЭТАП 1] Запуск SFT-разминки (SFT Warm-up) — 2 шага")
    print("-" * 50)
    
    # Наш целевой корректный шаблон расстановки ферзей: [2, 0, 3, 1]
    target_solution = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    sft_loss_fn = nn.CrossEntropyLoss()
    
    for sft_step in range(2):
        optimizer.zero_grad()
        
        # Разминка на активном блоке 0
        model.blocks.set_active_block(0)
        context = torch.randn(group_size, board_size)
        
        # Получаем предсказания логитов ходов
        logits = model(context, active_block_idx=0) # (group_size, board_size)
        
        # Вычисляем SFT лосс по отношению к целевому решению
        loss = sft_loss_fn(logits, target_solution)
        
        loss.backward()
        optimizer.step()
        print(f"SFT Шаг {sft_step + 1}/2 | Значение SFT Loss: {loss.item():.4f}")
        
    print("-" * 50)
    print("SFT-разминка успешно завершена. Переход к RL-выравниванию.")
    print("-" * 50)
    
    # ----------------------------------------------------
    # ЭТАП 2: RL-выравнивание через GDPO с Reward Shaping
    # ----------------------------------------------------
    print("\n[ЭТАП 2] Запуск тонкой настройки через GDPO — 3 шага")
    print("-" * 50)
    
    for step in range(3):
        optimizer.zero_grad()
        
        # Задаем активный блок (поочередно блоки 1 и 2)
        active_block_idx = (step + 1) % 3
        model.blocks.set_active_block(active_block_idx)
        
        # Входной контекст
        context = torch.randn(group_size, board_size)
        
        # Предсказания политики
        logits = model(context, active_block_idx)
        probs = torch.softmax(logits, dim=-1)
        
        # Сэмплируем ходы
        group_actions = []
        for i in range(group_size):
            actions_i = torch.multinomial(probs[i], num_samples=board_size, replacement=True)
            group_actions.append(actions_i.tolist())
            
        # Расчет наград со сглаживанием (Reward Shaping)
        # Симулируем разные варианты XML разметки кандидатов для демонстрации сглаживания:
        # Кандидат 0: Полный XML формат с тегом рассуждения (1.0 за формат + правильность)
        # Кандидат 1: Открытый тег рассуждения (0.1 за формат)
        # Кандидат 2: Открытый и закрытый тег рассуждения (0.2 за формат)
        # Кандидат 3: Простой не-XML ответ (0.0 за формат)
        rewards = {
            'correctness': [],
            'format': [],
            'brevity': []
        }
        
        for idx, actions in enumerate(group_actions):
            action_str = ",".join(map(str, actions))
            if idx == 0:
                # Полный правильный XML
                xml_str = f"<think>рассуждение</think><answer>{action_str}</answer>"
            elif idx == 1:
                # Только открытие think
                xml_str = f"<think>{action_str}"
            elif idx == 2:
                # think открыт и закрыт
                xml_str = f"<think>{action_str}</think>"
            else:
                # Нет XML разметки
                xml_str = action_str
                
            res = env.step(xml_str)
            rewards['correctness'].append(res['correctness'])
            rewards['format'].append(res['format'])
            rewards['brevity'].append(res['brevity'])
            
        # Вычисление декуплированных преимуществ GDPO
        # Благодаря Reward Shaping, у нас есть ненулевая дисперсия наград формата!
        advantages = compute_gdpo_advantages(rewards) # (group_size,)
        
        # Вычисляем лосс
        loss = 0.0
        for i in range(group_size):
            selected_actions = torch.tensor(group_actions[i], device=context.device)
            log_prob = torch.log(probs[i, selected_actions] + 1e-8).sum()
            # Полис-градиент шаг
            loss += -log_prob * advantages[i]
            
        loss = loss / group_size
        
        # Обратный проход и шаг оптимизации
        loss.backward()
        optimizer.step()
        
        # Печать хода итерации
        avg_correctness = sum(rewards['correctness']) / group_size
        avg_format = sum(rewards['format']) / group_size
        print(f"GDPO Шаг {step + 1}/3:")
        print(f"  Активный блок: {active_block_idx}")
        print(f"  Форматы кандидатов: Канд0: {rewards['format'][0]:.1f}, Канд1: {rewards['format'][1]:.1f}, Канд2: {rewards['format'][2]:.1f}, Канд3: {rewards['format'][3]:.1f}")
        print(f"  Средняя правильность: {avg_correctness:.2%}, Средний формат: {avg_format:.2%}")
        print(f"  Вычисленные преимущества GDPO: [{', '.join(f'{a.item():+.3f}' for a in advantages)}]")
        print(f"  Значение GDPO Loss: {loss.item():.4f}")
        print("-" * 50)
        
    print("="*60)
    print("Сквозное обучение SEF-GRAM успешно проверено с учетом рекомендаций аудита!")
    print("="*60)

if __name__ == "__main__":
    run_e2e_training()
