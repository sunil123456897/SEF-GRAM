import torch
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
from sef_gram.model import EFLAWorldModel
from sef_gram.environment import SokobanGenerator
from sef_gram.optimization import MuonWithAuxAdam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WorldModelSokoban(EFLAWorldModel):
    """
    Полная модель мира SEF-GRAM (VJEPA + EFLA + Decoder) для Sokoban.
    Унаследована от канонической EFLAWorldModel.
    """
    def __init__(self, input_dim=29, latent_dim=16, num_actions=4):
        super().__init__(input_dim=input_dim, latent_dim=latent_dim, output_dim=4, num_actions=num_actions)


def collect_sokoban_dynamics_data(num_scenes=20, traj_per_scene=20, traj_len=25):
    print(f"Сбор данных переходов Sokoban (5x5) в {num_scenes} конфигурациях...")
    generator = SokobanGenerator()
    data = []
    
    for scene_id in range(1, num_scenes + 1):
        walls, (ax, ay), (bx, by) = generator.generate(seed=scene_id)
        
        maze_map = np.zeros((5, 5))
        for wx, wy in walls:
            maze_map[wx, wy] = 1.0
        maze_map_flat = maze_map.flatten() # (25,)
        
        for traj_id in range(traj_per_scene):
            # Начинаем со стартовой позиции сцены
            x, y = ax, ay
            box_x, box_y = bx, by
            
            for step in range(traj_len):
                action = random.randint(0, 3)
                nx, ny = x, y
                n_box_x, n_box_y = box_x, box_y
                
                # Вычисляем смещение агента
                if action == 0:    ny = y - 1
                elif action == 1:  ny = y + 1
                elif action == 2:  nx = x - 1
                elif action == 3:  nx = x + 1
                
                # Физика Sokoban
                if (nx, ny) in walls:
                    # Столкновение со стеной
                    nx, ny = x, y
                elif (nx, ny) == (box_x, box_y):
                    # Агент толкает ящик
                    nnx, nny = box_x + (nx - x), box_y + (ny - y)
                    if (nnx, nny) not in walls:
                        # Успешный толчок ящика
                        n_box_x, n_box_y = nnx, nny
                    else:
                        # За ящиком стена - движение заблокировано
                        nx, ny = x, y
                
                # Записываем переход
                state = np.concatenate([[x, y, box_x, box_y], maze_map_flat])
                next_targets = np.array([nx, ny, n_box_x, n_box_y], dtype=np.float32)
                
                data.append((state, action, next_targets))
                
                x, y = nx, ny
                box_x, box_y = n_box_x, n_box_y
                
    print(f"Всего собрано {len(data)} шагов динамики Sokoban.")
    return data


def train_sokoban_world_model():
    print("="*80)
    print("ЭТАП 1: OOD DYNAMICS-ONLY ПРЕДОБУЧЕНИЕ МОДЕЛИ МИРА SOKOBAN (VJEPA+EFLA+PoE)")
    print(f"Используемое устройство: {device}")
    print("="*80)
    
    dataset = collect_sokoban_dynamics_data(num_scenes=20, traj_per_scene=20, traj_len=25)
    model = WorldModelSokoban(input_dim=29, latent_dim=16, num_actions=4).to(device)
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
    
    print("\nЗапуск обучения динамике Sokoban (150 эпох)...")
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
            
            # 1. Лосс автокодирования текущего состояния (восстановление ax, ay, bx, by)
            z_t = model.encoder.sample(mu_t, logvar_t)
            pred_current_coords = model.decoder(z_t)
            ae_loss = loss_fn(pred_current_coords, b_state[:, :4])
            
            # 2. Лосс физики переходов
            dyn_loss = loss_fn(pred_coords, b_next_coords)
            
            # 3. Метрическая регуляризация скрытого пространства
            dist_latent_t = torch.cdist(z_t, z_t, p=2)
            dist_coords_t = torch.cdist(b_state[:, :4], b_state[:, :4], p=2)
            metric_loss_t = torch.mean((dist_latent_t - dist_coords_t) ** 2)
            
            dist_latent_next = torch.cdist(z_next, z_next, p=2)
            dist_coords_next = torch.cdist(b_next_coords, b_next_coords, p=2)
            metric_loss_next = torch.mean((dist_latent_next - dist_coords_next) ** 2)
            
            metric_loss = 0.5 * (metric_loss_t + metric_loss_next)
            
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
    print(f"Обучение Sokoban успешно завершено за {time.time() - start_time:.2f} сек.")
    
    torch.save(model.state_dict(), "world_model_sokoban.pt")
    print("Веса модели Sokoban сохранены в 'world_model_sokoban.pt'")
    print("="*80)

if __name__ == "__main__":
    train_sokoban_world_model()
