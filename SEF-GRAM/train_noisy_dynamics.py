import torch
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
from sef_gram.model import EFLAWorldModel
from sef_gram.environment import MazeGenerator
from sef_gram.optimization import MuonWithAuxAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WorldModelNoisyDynamics(EFLAWorldModel):
    """
    Модель мира с шумоподавлением (входная размерность 112: 102 полезных признака + 10 дистракторов).
    Унаследована от канонической EFLAWorldModel.
    """
    def __init__(self, input_dim=112, latent_dim=16, num_actions=4):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, output_dim=2, num_actions=num_actions)


def collect_noisy_dynamics_data(num_mazes=20, traj_per_maze=15, traj_len=30):
    print(f"Сбор зашумленных данных переходов (10 дистракторов) в {num_mazes} лабиринтах...")
    generator = MazeGenerator()
    data = []
    
    for maze_id in range(1, num_mazes + 1):
        walls = generator.generate(seed=maze_id)
        maze_map = np.zeros((10, 10))
        for wx, wy in walls:
            maze_map[wx, wy] = 1.0
        maze_map_flat = maze_map.flatten()
        
        for traj_id in range(traj_per_maze):
            while True:
                x = random.randint(0, 9)
                y = random.randint(0, 9)
                if (x, y) not in walls:
                    break
                    
            for step in range(traj_len):
                action = random.randint(0, 3)
                nx, ny = x, y
                if action == 0:    ny = max(0, y - 1)
                elif action == 1:  ny = min(9, y + 1)
                elif action == 2:  nx = max(0, x - 1)
                elif action == 3:  nx = min(9, x + 1)
                if (nx, ny) in walls:
                    nx, ny = x, y
                    
                # Генерация 10 случайных дистракторов (шум)
                distractors = np.random.randn(10)
                
                # Полезные 102 признака + 10 шумных дистракторов = 112 признаков
                state = np.concatenate([[x, y], maze_map_flat, distractors])
                next_coords = np.array([nx, ny], dtype=np.float32)
                
                data.append((state, action, next_coords))
                x, y = nx, ny
                
    print(f"Всего собрано {len(data)} зашумленных шагов.")
    return data


def train_noisy_world_model():
    print("="*80)
    print("ЭТАП 1: ОБУЧЕНИЕ МОДЕЛИ МИРА С ПОДАВЛЕНИЕМ ШУМА (10 ДИСТРАКТОРОВ)")
    print(f"Используемое устройство: {device}")
    print("="*80)
    
    dataset = collect_noisy_dynamics_data(num_mazes=20, traj_per_maze=15, traj_len=30)
    model = WorldModelNoisyDynamics(input_dim=112, latent_dim=16, num_actions=4).to(device)
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.01, adamw_lr=1e-3)
    loss_fn = nn.MSELoss()
    
    states = torch.tensor([x[0] for x in dataset], dtype=torch.float32)
    actions = torch.tensor([x[1] for x in dataset], dtype=torch.long)
    next_coords_target = torch.tensor([x[2] for x in dataset], dtype=torch.float32)
    
    actions_onehot = torch.zeros(len(dataset), 4)
    actions_onehot.scatter_(1, actions.unsqueeze(1), 1.0)
    
    batch_size = 256
    dataset_size = len(dataset)
    num_epochs = 150
    
    print("\nЗапуск обучения динамике и подавлению шума (150 эпох)...")
    print("-" * 60)
    
    start_time = time.time()
    for epoch in range(num_epochs):
        indices = torch.randperm(dataset_size)
        epoch_losses = []
        epoch_ae_losses = []
        epoch_dyn_losses = []
        epoch_metric_losses = []
        
        for start_idx in range(0, dataset_size, batch_size):
            optimizer.zero_grad()
            batch_indices = indices[start_idx : start_idx + batch_size]
            
            b_state = states[batch_indices].to(device)
            b_action_onehot = actions_onehot[batch_indices].to(device)
            b_next_coords = next_coords_target[batch_indices].to(device)
            
            b_s_prev = torch.zeros(b_state.shape[0], 16, 16, device=device)
            
            # Прямой проход
            pred_coords, z_next, _, mu_t, logvar_t = model.forward_transition(b_state, b_action_onehot, b_s_prev)
            
            # 1. Лосс автокодирования текущих координат (декодер восстанавливает ТОЛЬКО координаты, игнорируя дистракторы!)
            z_t = model.encoder.sample(mu_t, logvar_t)
            pred_current_coords = model.decoder(z_t)
            ae_loss = loss_fn(pred_current_coords, b_state[:, :2])
            
            # 2. Лосс физики переходов
            dyn_loss = loss_fn(pred_coords, b_next_coords)
            
            # 3. Метрическая регуляризация скрытого пространства
            dist_latent_t = torch.cdist(z_t, z_t, p=2)
            dist_coords_t = torch.cdist(b_state[:, :2], b_state[:, :2], p=2)
            metric_loss_t = torch.mean((dist_latent_t - dist_coords_t) ** 2)
            
            dist_latent_next = torch.cdist(z_next, z_next, p=2)
            dist_coords_next = torch.cdist(b_next_coords, b_next_coords, p=2)
            metric_loss_next = torch.mean((dist_latent_next - dist_coords_next) ** 2)
            
            metric_loss = 0.5 * (metric_loss_t + metric_loss_next)
            
            # Совместный лосс
            loss = ae_loss + dyn_loss + 0.5 * metric_loss
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_ae_losses.append(ae_loss.item())
            epoch_dyn_losses.append(dyn_loss.item())
            epoch_metric_losses.append(metric_loss.item())
            
        if (epoch + 1) % 15 == 0 or epoch == 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_ae = sum(epoch_ae_losses) / len(epoch_ae_losses)
            avg_dyn = sum(epoch_dyn_losses) / len(epoch_dyn_losses)
            avg_metric = sum(epoch_metric_losses) / len(epoch_metric_losses)
            print(f"Эпоха {epoch + 1:03d}/150 | Общий Loss: {avg_loss:.4f} | AE Loss: {avg_ae:.4f} | Physics Loss: {avg_dyn:.4f} | Metric Loss: {avg_metric:.4f}")
            
    print("-" * 60)
    print(f"Обучение успешно завершено за {time.time() - start_time:.2f} сек.")
    
    torch.save(model.state_dict(), "world_model_noisy_dynamics.pt")
    print("Веса модели сохранены в 'world_model_noisy_dynamics.pt'")
    print("="*80)

if __name__ == "__main__":
    train_noisy_world_model()
