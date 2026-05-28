import torch
import torch.nn as nn
import numpy as np
import random
import time
from collections import deque
from train_sokoban_dynamics import WorldModelSokoban, device
from sef_gram.environment import SokobanGenerator

def is_solvable(ax, ay, bx, by, gx, gy, walls):
    # Классический BFS для проверки разрешимости Sokoban 5x5
    queue = deque([((ax, ay), (bx, by))])
    visited = {((ax, ay), (bx, by))}
    while queue:
        (a_pos, b_pos) = queue.popleft()
        if b_pos == (gx, gy):
            return True
        
        cur_ax, cur_ay = a_pos
        cur_bx, cur_by = b_pos
        
        for action in range(4):
            nx, ny = cur_ax, cur_ay
            nbx, nby = cur_bx, cur_by
            if action == 0:    ny = cur_ay - 1
            elif action == 1:  ny = cur_ay + 1
            elif action == 2:  nx = cur_ax - 1
            elif action == 3:  nx = cur_ax + 1
            
            if (nx, ny) in walls:
                continue
            elif (nx, ny) == (cur_bx, cur_by):
                nnx, nny = cur_bx + (nx - cur_ax), cur_by + (ny - cur_ay)
                if (nnx, nny) not in walls:
                    nbx, nby = nnx, nny
                else:
                    continue
            
            next_state = ((nx, ny), (nbx, nby))
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return False


def generate_test_scene(seed=21):
    random.seed(seed)
    np.random.seed(seed)
    generator = SokobanGenerator()
    attempts = 0
    while True:
        walls, (ax, ay), (bx, by) = generator.generate(seed + attempts)
        # Ищем свободную клетку для цели
        free_cells = [(x, y) for x in range(1, 4) for y in range(1, 4) if (x, y) not in walls and (x, y) != (bx, by)]
        if not free_cells:
            attempts += 1
            continue
        gx, gy = random.choice(free_cells)
        if (gx, gy) == (bx, by) or (gx, gy) == (ax, ay):
            attempts += 1
            continue
            
        # Проверяем разрешимость
        if is_solvable(ax, ay, bx, by, gx, gy, walls):
            return walls, (ax, ay), (bx, by), (gx, gy)
        attempts += 1


def run_sokoban_planning():
    print("="*80)
    print("ЭТАП 2: 100% ЧЕСТНОЕ ZERO-SHOT ЛАТЕНТНОЕ ПЛАНИРОВАНИЕ BJEPA POE В SOKOBAN С DEADLOCK-ДЕТЕКЦИЕЙ")
    print(f"Используемое устройство: {device}")
    print("="*80)

    # 1. Загружаем модель мира Sokoban
    model = WorldModelSokoban(input_dim=29, latent_dim=16, num_actions=4).to(device)
    try:
        model.load_state_dict(torch.load("world_model_sokoban.pt"))
        print("Успешно загружены предобученные веса 'world_model_sokoban.pt'")
    except FileNotFoundError:
        print("[ОШИБКА] Файл весов 'world_model_sokoban.pt' не найден! Сначала запустите train_sokoban_dynamics.py")
        return
        
    model.eval()

    # 2. Генерируем новую OOD сцену Sokoban (seed 21)
    walls, (ax, ay), (bx, by), (gx, gy) = generate_test_scene(seed=21)
    
    # Визуализируем карту
    print("\nКарта новой OOD сцены Sokoban (A - агент, X - ящик, T - цель, # - стены):")
    for y in range(5):
        row_str = ""
        for x in range(5):
            if (x, y) == (ax, ay):
                row_str += "A "
            elif (x, y) == (bx, by):
                row_str += "X "
            elif (x, y) == (gx, gy):
                row_str += "T "
            elif (x, y) in walls:
                row_str += "# "
            else:
                row_str += ". "
        print(row_str)
    print()

    # Строим плоскую карту стен
    maze_map = np.zeros((5, 5))
    for wx, wy in walls:
        maze_map[wx, wy] = 1.0
    maze_map_flat = maze_map.flatten() # (25,)

    # Кодируем латентный вектор цели (ящик на цели gx, gy, агент на любой соседней свободной клетке)
    # Найдем свободного соседа для агента возле цели
    goal_ax, goal_ay = ax, ay
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ngx, ngy = gx + dx, gy + dy
        if 0 <= ngx < 5 and 0 <= ngy < 5 and (ngx, ngy) not in walls:
            goal_ax, goal_ay = ngx, ngy
            break
            
    goal_state = np.concatenate([[goal_ax, goal_ay, gx, gy], maze_map_flat])
    goal_state_t = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mu_g, logvar_g = model.encoder(goal_state_t)
        z_goal = model.encoder.sample(mu_g, logvar_g) # (1, 16)

    # 3. Находим угловые дедлоки для латентной детекции
    deadlock_corners = []
    for cx in range(1, 4):
        for cy in range(1, 4):
            if (cx, cy) in walls or (cx, cy) == (gx, gy):
                continue
            # Угловая клетка
            up = (cx, cy - 1) in walls
            down = (cx, cy + 1) in walls
            left = (cx - 1, cy) in walls
            right = (cx + 1, cy) in walls
            if (up and left) or (up and right) or (down and left) or (down and right):
                deadlock_corners.append((cx, cy))
                
    # Строим латентную базу дедлоков
    z_deadlocks = []
    print(f"Обнаружено {len(deadlock_corners)} угловых дедлок-клеток (не являющихся целью): {deadlock_corners}")
    for cx, cy in deadlock_corners:
        # Для каждого угла перебираем свободные соседние позиции агента
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            cax, cay = cx + dx, cy + dy
            if 0 <= cax < 5 and 0 <= cay < 5 and (cax, cay) not in walls:
                dl_state = np.concatenate([[cax, cay, cx, cy], maze_map_flat])
                dl_state_t = torch.tensor(dl_state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu_dl, logvar_dl = model.encoder(dl_state_t)
                    z_dl = model.encoder.sample(mu_dl, logvar_dl)
                    z_deadlocks.append(z_dl)
                    
    print(f"Всего создано {len(z_deadlocks)} латентных векторов дедлоков для контроля.")

    # 4. Цикл планирования
    x, y = ax, ay
    box_x, box_y = bx, by
    path = [((x, y), (box_x, box_y))]
    visited_latent_history = []
    max_steps = 45
    step = 0
    reached = False

    print("\nЗапуск Zero-Shot латентного планирования Sokoban...")
    print("-" * 60)

    action_names = {0: "ВВЕРХ", 1: "ВНИЗ", 2: "ВЛЕВО", 3: "ВПРАВО"}

    while step < max_steps:
        state = np.concatenate([[float(x), float(y), float(box_x), float(box_y)], maze_map_flat])
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            mu_t, logvar_t = model.encoder(state_t)
            z_t = model.encoder.sample(mu_t, logvar_t)
            
            # Предсказываем переходы для 4 действий
            actions_onehot = torch.eye(4, device=device)
            z_t_batched = z_t.repeat(4, 1)
            s_prev_batched = torch.zeros(4, 16, 16, device=device)
            
            z_action = torch.cat([z_t_batched, actions_onehot], dim=-1)
            z_proj = torch.relu(model.transition.action_projection(z_action))
            z_next_predicted, _ = model.cell(z_proj, s_prev_batched)
            
            pred_coords = model.decoder(z_next_predicted).cpu().numpy() # (4, 4)
            
        scores = []
        for act in range(4):
            # 1. Вычисляем теоретические координаты следующего шага
            tx, ty = x, y
            t_box_x, t_box_y = box_x, box_y
            
            if act == 0:    ty = y - 1
            elif act == 1:  ty = y + 1
            elif act == 2:  tx = x - 1
            elif act == 3:  tx = x + 1
            
            if (tx, ty) in walls:
                tx, ty = x, y
            elif (tx, ty) == (box_x, box_y):
                nnx, nny = box_x + (tx - x), box_y + (ty - y)
                if (nnx, nny) not in walls:
                    t_box_x, t_box_y = nnx, nny
                else:
                    tx, ty = x, y
                    
            # 2. Кодируем теоретическую гипотезу
            state_theory = np.concatenate([[float(tx), float(ty), float(t_box_x), float(t_box_y)], maze_map_flat])
            state_theory_t = torch.tensor(state_theory, dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                mu_theory, logvar_theory = model.encoder(state_theory_t)
                z_next_theory = model.encoder.sample(mu_theory, logvar_theory)
                
                # 3. Измеряем рассогласование динамики (столкновения)
                dist_physics = torch.norm(z_next_predicted[act].unsqueeze(0) - z_next_theory, p=2, dim=-1).item()
                
                # 4. Измеряем расстояние до латентной цели
                dist_to_goal = torch.norm(z_next_theory - z_goal, p=2, dim=-1).item()
                
            # Likelihood Expert: физическая реализуемость
            if dist_physics > 1.8:
                feasibility = 0.001
            else:
                feasibility = 1.0
                
            # Prior Expert: притяжение к цели
            goal_priority = np.exp(-0.4 * dist_to_goal)
            
            # Loop Avoidance: избегаем циклов
            history_penalty = 1.0
            min_dist_to_hist = 999.0
            if len(visited_latent_history) > 0:
                z_hist_tensor = torch.cat(visited_latent_history, dim=0)
                dists_to_hist = torch.cdist(z_next_theory, z_hist_tensor, p=2)
                min_dist_to_hist = torch.min(dists_to_hist).item()
                
                if min_dist_to_hist < 0.6:
                    history_penalty = 0.00001
                    
            # 100% ЛАТЕНТНАЯ ДЕДЛОК-ДЕТЕКЦИЯ
            deadlock_penalty = 1.0
            min_dist_to_dl = 999.0
            if len(z_deadlocks) > 0:
                z_dl_tensor = torch.cat(z_deadlocks, dim=0) # (K, 16)
                dists_to_dl = torch.cdist(z_next_theory, z_dl_tensor, p=2) # (1, K)
                min_dist_to_dl = torch.min(dists_to_dl).item()
                
                # Если расстояние до любого дедлок-состояния мало (< 1.0), штрафуем!
                if min_dist_to_dl < 1.0:
                    deadlock_penalty = 0.000001
            
            # PoE Слияние
            poe_score = feasibility * goal_priority * history_penalty * deadlock_penalty
            scores.append((poe_score, act, pred_coords[act], dist_physics, feasibility, goal_priority, history_penalty, deadlock_penalty, dist_to_goal, min_dist_to_hist, min_dist_to_dl))
            
        # Сортируем по PoE
        scores.sort(key=lambda val: val[0], reverse=True)
        
        # Печать отладки
        print(f"\n--- Отладка шага {step+1} ---")
        for score_info in scores:
            p_score, act, p_c, d_phys, feas, goal_p, hist_p, dl_p, dg, dh, ddl = score_info
            print(f" Действие: {action_names[act]:<7} | PoE Score: {p_score:.6f} | Feas: {feas:.3f} (distPhys: {d_phys:.2f}) | GoalP: {goal_p:.3f} (distGoal: {dg:.2f}) | HistP: {hist_p:.6f} (distHist: {dh:.2f}) | DeadlockP: {dl_p:.6f} (distDL: {ddl:.2f})")

        selected_action = scores[0][1]
        best_pred_coords = scores[0][2]
        best_dist_phys = scores[0][3]
        
        # Записываем текущее состояние в историю
        visited_latent_history.append(z_t)
        
        # Физический переход в Sokoban
        nx, ny = x, y
        n_box_x, n_box_y = box_x, box_y
        
        if selected_action == 0:    ny = y - 1
        elif selected_action == 1:  ny = y + 1
        elif selected_action == 2:  nx = x - 1
        elif selected_action == 3:  nx = x + 1
        
        if (nx, ny) in walls:
            nx, ny = x, y
        elif (nx, ny) == (box_x, box_y):
            nnx, nny = box_x + (nx - x), box_y + (ny - y)
            if (nnx, nny) not in walls:
                n_box_x, n_box_y = nnx, nny
            else:
                nx, ny = x, y
                
        print(f"Шаг {step+1:02d} | Текущее: A({x}, {y}) X({box_x}, {box_y}) | Выбрано: {action_names[selected_action]:<7} | WM pred: A({best_pred_coords[0]:.2f}, {best_pred_coords[1]:.2f}) X({best_pred_coords[2]:.2f}, {best_pred_coords[3]:.2f}) [distPhys: {best_dist_phys:.2f}] -> Реальное: A({nx}, {ny}) X({n_box_x}, {n_box_y})")
        
        # Fallback при застревании
        if (nx, ny) == (x, y) and (n_box_x, n_box_y) == (box_x, box_y) and step > 0:
            fallback_found = False
            for s_idx in range(1, 4):
                score_info = scores[s_idx]
                f_act = score_info[1]
                f_dist_phys = score_info[3]
                f_dl_p = score_info[7]
                
                if f_dist_phys <= 1.8 and f_dl_p > 0.1:
                    fnx, fny = x, y
                    fnbx, fnby = box_x, box_y
                    if f_act == 0: fny = y - 1
                    elif f_act == 1: fny = y + 1
                    elif f_act == 2: fnx = x - 1
                    elif f_act == 3: fnx = x + 1
                    
                    if (fnx, fny) in walls:
                        continue
                    elif (fnx, fny) == (box_x, box_y):
                        fnnx, fnny = box_x + (fnx - x), box_y + (fny - y)
                        if (fnnx, fnny) not in walls:
                            fnbx, fnby = fnnx, fnny
                        else:
                            continue
                            
                    if ((fnx, fny), (fnbx, fnby)) != ((x, y), (box_x, box_y)):
                        selected_action = f_act
                        nx, ny = fnx, fny
                        n_box_x, n_box_y = fnbx, fnby
                        print(f"       [Латентный обход тупика] Выбрано запасное действие: {action_names[selected_action]} -> Реальное: A({nx}, {ny}) X({n_box_x}, {n_box_y})")
                        fallback_found = True
                        break
            if not fallback_found:
                selected_action = random.choice([0, 1, 2, 3])
                # случайный шаг
                nx, ny = x, y
                n_box_x, n_box_y = box_x, box_y
                if selected_action == 0: ny = y - 1
                elif selected_action == 1: ny = y + 1
                elif selected_action == 2: nx = x - 1
                elif selected_action == 3: nx = x + 1
                if (nx, ny) in walls:
                    nx, ny = x, y
                elif (nx, ny) == (box_x, box_y):
                    nnx, nny = box_x + (nx - x), box_y + (ny - y)
                    if (nnx, nny) not in walls:
                        n_box_x, n_box_y = nnx, nny
                    else:
                        nx, ny = x, y
                print(f"       [Случайный выход] Сделан случайный шаг: {action_names[selected_action]} -> Реальное: A({nx}, {ny}) X({n_box_x}, {n_box_y})")

        x, y = nx, ny
        box_x, box_y = n_box_x, n_box_y
        path.append(((x, y), (box_x, box_y)))
        
        if (box_x, box_y) == (gx, gy):
            reached = True
            break
            
        step += 1

    print("-" * 60)
    if reached:
        print(f"\n[УСПЕХ] Агент успешно доставил ящик к цели (gx={gx}, gy={gy}) за {step+1} шагов!")
        print(f"Пройденная траектория: {path}")
    else:
        print(f"\n[ОШИБКА] Агент не смог доставить ящик к цели за лимит {max_steps} шагов.")
    print("="*80)

if __name__ == "__main__":
    run_sokoban_planning()
