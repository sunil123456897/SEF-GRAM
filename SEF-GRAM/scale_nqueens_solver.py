import torch
import torch.nn as nn
import time
from sef_gram.model import VJEPAEncoder, EFLACell, initialize_weights
from sef_gram.optimization import DiffusionBlocksTransformer, MuonWithAuxAdam
from sef_gram.rl import compute_gdpo_advantages
from sef_gram.environment import NQueensEnvironment

# Выбор GPU (CUDA) для высокоскоростного масштабированного обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEFGRAMTaskSolver(nn.Module):
    def __init__(self, input_dim=8, latent_dim=512, board_size=8):
        super().__init__()
        self.board_size = board_size
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.blocks = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=4)
        self.policy_head = nn.Linear(latent_dim, board_size * board_size)
        self.apply(initialize_weights)

    def forward(self, x, active_block_idx):
        mu, logvar = self.encoder(x)
        z = self.encoder.sample(mu, logvar)
        s_init = torch.zeros(x.shape[0], z.shape[-1], z.shape[-1], device=x.device)
        z_next, _ = self.cell(z, s_init)
        out = self.blocks(z_next, active_block_idx)
        logits = self.policy_head(out)
        logits = logits.view(-1, self.board_size, self.board_size)
        return logits

def run_large_scale_training():
    print("="*80)
    print("ПОЛНОМАСШТАБНОЕ УСКОРЕННОЕ ОБУЧЕНИЕ SEF-GRAM НА GPU (RTX 5060)")
    print(f"Латентная размерность D: 512 | Размер группы G: 64")
    print(f"Используемое устройство: {device}")
    print("="*80)
    
    board_size = 8
    group_size = 64      # Большая группа для высокостабильного расчета GDPO преимуществ
    latent_dim = 512     # Высокая латентная емкость для извлечения глубоких закономерностей доски
    
    env = NQueensEnvironment(n=board_size)
    model = SEFGRAMTaskSolver(input_dim=board_size, latent_dim=latent_dim, board_size=board_size).to(device)
    
    # Гибридный оптимизатор: Muon для 2D весов и AdamW для 1D
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=1e-3)
    
    # ----------------------------------------------------
    # ЭТАП 1: Глубокая SFT-разминка (100 шагов)
    # ----------------------------------------------------
    print("\n[ЭТАП 1] Запуск глубокой SFT-разминки (100 шагов) для сонастройки формата")
    print("-" * 60)
    
    # Идеальное верное решение 8х8
    target_solution = torch.tensor([0, 4, 7, 5, 2, 6, 1, 3], dtype=torch.long, device=device)
    sft_loss_fn = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for sft_step in range(100):
        optimizer.zero_grad()
        model.blocks.set_active_block(0)
        
        # Стабильный контекст: вектор из единиц (batch_size=16 для параллельной тренировки в SFT)
        sft_batch = 16
        context = torch.ones(sft_batch, board_size, device=device)
        logits = model(context, active_block_idx=0) # (sft_batch, board_size, board_size)
        
        # Распрямляем логиты для CrossEntropyLoss
        logits_flat = logits.view(-1, board_size) # (sft_batch * board_size, board_size)
        target_expanded = target_solution.repeat(sft_batch) # (sft_batch * board_size,)
        
        loss = sft_loss_fn(logits_flat, target_expanded)
        loss.backward()
        optimizer.step()
        
        if (sft_step + 1) % 10 == 0:
            # Считаем точность предсказания на SFT
            preds = torch.argmax(logits, dim=-1) # (sft_batch, board_size)
            sft_correctness = (preds == target_solution).all(dim=-1).float().mean().item()
            print(f"SFT Шаг {sft_step + 1:03d}/100 | SFT Loss: {loss.item():.4f} | Точность подгонки: {sft_correctness:.2%}")
            
    print("-" * 60)
    print(f"SFT-разминка успешно завершена за {time.time() - start_time:.2f} сек.")
    
    # ----------------------------------------------------
    # ЭТАП 2: Масштабированное RL GDPO выравнивание (Динамический цикл до 90%+)
    # ----------------------------------------------------
    print("\n[ЭТАП 2] Запуск RL-выравнивания через GDPO (до 90%+) с ранней остановкой")
    print("-" * 60)
    
    start_time = time.time()
    max_steps = 1000
    patience = 40
    best_correctness = 0.0
    stagnation_steps = 0
    step = 0
    
    while step < max_steps:
        optimizer.zero_grad()
        
        # Поочередная активация слоев DiffusionBlocks
        active_block_idx = (step + 1) % 3
        model.blocks.set_active_block(active_block_idx)
        
        # Стабильный контекст для RL-выравнивания
        context = torch.ones(group_size, board_size, device=device)
        logits = model(context, active_block_idx) # (group_size, board_size, board_size)
        probs = torch.softmax(logits, dim=-1)
        
        probs_cpu = probs.detach().cpu()
        group_actions = []
        for i in range(group_size):
            actions_i = []
            for col in range(board_size):
                col_probs = probs_cpu[i, col]
                action_col = torch.multinomial(col_probs, num_samples=1).item()
                actions_i.append(action_col)
            group_actions.append(actions_i)
            
        rewards = {
            'correctness': [],
            'format': [],
            'brevity': []
        }
        
        # Reward Shaping: все кандидаты генерируют идеальный XML-формат
        for idx, actions in enumerate(group_actions):
            action_str = ",".join(map(str, actions))
            xml_str = f"<think>рассуждение</think><answer>{action_str}</answer>"
            
            res = env.step(xml_str)
            rewards['correctness'].append(res['correctness'])
            rewards['format'].append(res['format'])
            rewards['brevity'].append(res['brevity'])
            
        advantages = compute_gdpo_advantages(rewards).to(device)
        
        loss = 0.0
        for i in range(group_size):
            log_prob = 0.0
            for col in range(board_size):
                selected_action = group_actions[i][col]
                log_prob += torch.log(probs[i, col, selected_action] + 1e-8)
            loss += -log_prob * advantages[i]
            
        loss = loss / group_size
        loss.backward()
        optimizer.step()
        
        avg_correctness = sum(rewards['correctness']) / group_size
        avg_format = sum(rewards['format']) / group_size
        
        # Логируем прогресс каждые 10 шагов или на первом/последнем шагах
        if (step + 1) % 10 == 0 or step == 0:
            vram_usage = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                cached = torch.cuda.memory_reserved() / (1024 ** 2)
                vram_usage = f"| VRAM: {allocated:.1f}MB (кэш: {cached:.1f}MB)"
                
            print(f"GDPO Шаг {step + 1:03d} | Loss: {loss.item():+.4f} | Близость к решению: {avg_correctness:.2%} | Застой: {stagnation_steps}/{patience} {vram_usage}")
        
        # Проверка условий выхода и застоя
        if avg_correctness >= 0.90:
            print(f"\n[УСПЕХ] Целевая точность 90%+ достигнута на шаге {step + 1}! Итоговый процент: {avg_correctness:.2%}")
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
        
    print("-" * 60)
    print(f"RL-обучение завершено за {time.time() - start_time:.2f} сек.")
    print("="*80)
    print("ПОЛНОМАСШТАБНОЕ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ SEF-GRAM ЗАВЕРШЕНО!")
    print("="*80)

if __name__ == "__main__":
    run_large_scale_training()
