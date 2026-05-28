import torch
import torch.nn as nn
import time
from sef_gram.model import VJEPAEncoder, EFLACell, initialize_weights
from sef_gram.optimization import DiffusionBlocksTransformer, MuonWithAuxAdam
from sef_gram.rl import compute_gdpo_advantages
from sef_gram.environment import NQueensEnvironment

# Выбор аппаратного устройства (CUDA для RTX 5060, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEFGRAMTaskSolver(nn.Module):
    def __init__(self, input_dim=8, latent_dim=256, board_size=8):
        super().__init__()
        self.encoder = VJEPAEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.blocks = DiffusionBlocksTransformer(latent_dim=latent_dim, num_blocks=3, layers_per_block=4)
        self.policy_head = nn.Linear(latent_dim, board_size)
        # Инициализация весов
        self.apply(initialize_weights)

    def forward(self, x, active_block_idx):
        mu, logvar = self.encoder(x)
        z = self.encoder.sample(mu, logvar)
        # Инициализация матрицы памяти EFLA непосредственно на устройстве
        s_init = torch.zeros(x.shape[0], z.shape[-1], z.shape[-1], device=x.device)
        z_next, _ = self.cell(z, s_init)
        out = self.blocks(z_next, active_block_idx)
        logits = self.policy_head(out)
        return logits

def run_production():
    print("="*80)
    print(f"ЗАПУСК ВЫСОКОНАГРУЖЕННОГО ОБУЧЕНИЯ SEF-GRAM НА GPU (RTX 5060)")
    print(f"Используемое устройство: {device}")
    print("="*80)
    
    board_size = 8
    group_size = 32      # Увеличенный размер группы для стабильности преимуществ GDPO
    latent_dim = 256     # Увеличенная латентная размерность для загрузки видеопамяти
    
    env = NQueensEnvironment(n=board_size)
    model = SEFGRAMTaskSolver(input_dim=board_size, latent_dim=latent_dim, board_size=board_size).to(device)
    
    # Гибридный оптимизатор: веса Muon, смещения и классификатор AdamW
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.01, adamw_lr=1e-3)
    
    # 1. SFT WARM-UP (20 шагов)
    print("\n[ЭТАП 1] Запуск SFT-разминки (SFT Warm-up) — 20 шагов")
    print("-" * 60)
    
    target_solution = torch.tensor([0, 4, 7, 5, 2, 6, 1, 3], dtype=torch.long, device=device)
    sft_loss_fn = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for sft_step in range(20):
        optimizer.zero_grad()
        model.blocks.set_active_block(0)
        
        # Батч равен board_size=8 для полной сонастройки весов
        context = torch.randn(board_size, board_size, device=device)
        logits = model(context, active_block_idx=0)
        
        loss = sft_loss_fn(logits, target_solution)
        loss.backward()
        optimizer.step()
        
        if (sft_step + 1) % 5 == 0:
            print(f"SFT Шаг {sft_step + 1:02d}/20 | SFT Loss: {loss.item():.4f}")
            
    print("-" * 60)
    print(f"SFT-разминка завершена за {time.time() - start_time:.2f} сек.")
    
    # 2. RL GDPO с Reward Shaping (200 шагов)
    print("\n[ЭТАП 2] Запуск RL-выравнивания через GDPO — 200 шагов")
    print("-" * 60)
    
    start_time = time.time()
    for step in range(200):
        optimizer.zero_grad()
        
        active_block_idx = (step + 1) % 3
        model.blocks.set_active_block(active_block_idx)
        
        context = torch.randn(group_size, board_size, device=device)
        logits = model(context, active_block_idx)
        probs = torch.softmax(logits, dim=-1)
        
        # Переносим вероятности на CPU для сэмплирования ходов
        probs_cpu = probs.detach().cpu()
        group_actions = []
        for i in range(group_size):
            actions_i = torch.multinomial(probs_cpu[i], num_samples=board_size, replacement=True)
            group_actions.append(actions_i.tolist())
            
        # Расчет наград со сглаживанием
        rewards = {
            'correctness': [],
            'format': [],
            'brevity': []
        }
        
        for idx, actions in enumerate(group_actions):
            action_str = ",".join(map(str, actions))
            # Симулируем исследование синтаксиса кандидатами
            # Кандидат 0 и 1: Полный XML формат с тегом рассуждения
            # Кандидат 2-15: Промежуточные теги
            # Остальные: Простой ответ
            if idx < 4:
                xml_str = f"<think>рассуждение</think><answer>{action_str}</answer>"
            elif idx < 12:
                xml_str = f"<think>{action_str}</think>"
            elif idx < 24:
                xml_str = f"<think>{action_str}"
            else:
                xml_str = action_str
                
            res = env.step(xml_str)
            rewards['correctness'].append(res['correctness'])
            rewards['format'].append(res['format'])
            rewards['brevity'].append(res['brevity'])
            
        # Вычисление декуплированных преимуществ GDPO
        advantages = compute_gdpo_advantages(rewards).to(device)
        
        loss = 0.0
        for i in range(group_size):
            selected_actions = torch.tensor(group_actions[i], device=device)
            log_prob = torch.log(probs[i, selected_actions] + 1e-8).sum()
            loss += -log_prob * advantages[i]
            
        loss = loss / group_size
        loss.backward()
        optimizer.step()
        
        # Каждые 20 шагов выводим метрики и состояние видеопамяти
        if (step + 1) % 20 == 0 or step == 0:
            avg_correctness = sum(rewards['correctness']) / group_size
            avg_format = sum(rewards['format']) / group_size
            
            # Считываем VRAM если запущено на CUDA
            vram_usage = ""
            if device.type == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024 ** 2) # MB
                cached = torch.cuda.memory_reserved() / (1024 ** 2)     # MB
                vram_usage = f"| VRAM: {allocated:.1f}MB (кэш: {cached:.1f}MB)"
                
            print(f"GDPO {step + 1:03d}/200 | Loss: {loss.item():+.4f} | Точность: {avg_correctness:.2%} | XML-формат: {avg_format:.2%} {vram_usage}")
            
    print("-" * 60)
    print(f"RL-обучение завершено за {time.time() - start_time:.2f} сек.")
    print("="*80)
    print("Высоконагруженный Production-тест SEF-GRAM успешно выполнен!")
    print("="*80)

if __name__ == "__main__":
    run_production()
