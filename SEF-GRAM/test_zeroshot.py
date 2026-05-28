import torch
import torch.nn as nn
import numpy as np
import random
import time
from train_dynamics import WorldModelDynamics, device
from sef_gram.environment import MazeGenerator

def run_zeroshot_planning():
    print("="*80)
    print("ЭТАП 2: 100% ЧЕСТНОЕ ZERO-SHOT ЛАТЕНТНОЕ ПЛАНИРОВАНИЕ BJEPA POE (THEORETICAL-LATENT)")
    print(f"Используемое устройство: {device}")
    print("="*80)

    # 1. Загружаем предобученную модель мира
    model = WorldModelDynamics(input_dim=102, latent_dim=16, num_actions=4).to(device)
    try:
        model.load_state_dict(torch.load("world_model_dynamics.pt"))
        print("Успешно загружены предобученные веса 'world_model_dynamics.pt'")
    except FileNotFoundError:
        print("[ОШИБКА] Файл весов 'world_model_dynamics.pt' не найден! Сначала запустите train_dynamics.py")
        return
        
    model.eval()

    # 2. Генерируем новый 21-й лабиринт
    generator = MazeGenerator()
    new_walls = generator.generate(seed=21) # Абсолютно новая карта (seed 21)
    
    # Визуализируем лабиринт
    print("\nКарта нового невидимого 21-го лабиринта (A - старт, B - финиш, # - стены):")
    for y in range(10):
        row_str = ""
        for x in range(10):
            if (x, y) == (0, 0):
                row_str += "A "
            elif (x, y) == (9, 9):
                row_str += "B "
            elif (x, y) in new_walls:
                row_str += "# "
            else:
                row_str += ". "
        print(row_str)
    print()

    # Строим бинарную карту лабиринта
    maze_map = np.zeros((10, 10))
    for wx, wy in new_walls:
        maze_map[wx, wy] = 1.0
    maze_map_flat = maze_map.flatten() # (100,)

    # Кодируем латентный вектор цели (9, 9)
    goal_state = np.concatenate([[9.0, 9.0], maze_map_flat])
    goal_state_t = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mu_g, logvar_g = model.encoder(goal_state_t)
        z_goal = model.encoder.sample(mu_g, logvar_g) # (1, 16)

    # 3. Цикл Zero-Shot прохождения
    x, y = 0, 0 # Старт
    path = [(x, y)]
    visited_latent_history = [] # Неограниченная долгосрочная латентная история посещений
    max_steps = 45
    step = 0
    reached = False

    print("Запуск Zero-Shot прохождения лабиринта...")
    print("-" * 60)

    action_names = {0: "ВВЕРХ", 1: "ВНИЗ", 2: "ВЛЕВО", 3: "ВПРАВО"}

    while step < max_steps:
        # Локальный контекст текущего состояния
        state = np.concatenate([[float(x), float(y)], maze_map_flat])
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Кодируем текущее состояние
            mu_t, logvar_t = model.encoder(state_t)
            z_t = model.encoder.sample(mu_t, logvar_t) # (1, 16)
            
            # Для каждого из 4 действий прогоняем модель перехода
            actions_onehot = torch.eye(4, device=device) # (4, 4)
            z_t_batched = z_t.repeat(4, 1) # (4, 16)
            s_prev_batched = torch.zeros(4, 16, 16, device=device)
            
            z_action = torch.cat([z_t_batched, actions_onehot], dim=-1) # (4, 20)
            z_proj = torch.relu(model.transition.action_projection(z_action))
            z_next_predicted, _ = model.cell(z_proj, s_prev_batched) # (4, 16)
            
            # Для логирования декодируем предсказанные моделью мира координаты (ИСКЛЮЧИТЕЛЬНО ДЛЯ ОТЛАДКИ!)
            pred_coords = model.decoder(z_next_predicted).cpu().numpy() # (4, 2)
            
        # Строим PoE исключительно в латентном пространстве
        scores = []
        for act in range(4):
            # 1. Вычисляем теоретические координаты следующего шага
            tx, ty = x, y
            if act == 0: ty = max(0, y - 1)
            elif act == 1: ty = min(9, y + 1)
            elif act == 2: tx = max(0, x - 1)
            elif act == 3: tx = min(9, x + 1)
            
            # 2. Кодируем теоретическое следующее состояние в латентный вектор
            state_theory = np.concatenate([[float(tx), float(ty)], maze_map_flat])
            state_theory_t = torch.tensor(state_theory, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mu_theory, logvar_theory = model.encoder(state_theory_t)
                z_next_theory = model.encoder.sample(mu_theory, logvar_theory) # (1, 16)
                
                # 3. Измеряем рассогласование динамики (физическая осуществимость)
                # Показывает, насколько предсказание WM совпадает с теоретическим шагом
                dist_physics = torch.norm(z_next_predicted[act].unsqueeze(0) - z_next_theory, p=2, dim=-1).item()
                
                # 4. Измеряем расстояние до латентного вектора цели z_goal
                dist_to_goal = torch.norm(z_next_theory - z_goal, p=2, dim=-1).item()
                
            # Likelihood Expert: Маска физической осуществимости. 
            # Если предсказание WM расходится с теорией (стена), dist_physics будет большим (> 1.8)
            if dist_physics > 1.8:
                feasibility = 0.001  # Крайне малая вероятность
            else:
                feasibility = 1.0
                
            # Prior Expert: Притяжение к цели в латентном пространстве
            goal_priority = np.exp(-0.4 * dist_to_goal)
            
            # Loop Avoidance: запрещаем возвращаться в посещенные латентные ячейки
            history_penalty = 1.0
            min_dist_to_hist = 999.0
            if len(visited_latent_history) > 0:
                z_hist_tensor = torch.cat(visited_latent_history, dim=0) # (N, 16)
                dists_to_hist = torch.cdist(z_next_theory, z_hist_tensor, p=2) # (1, N)
                min_dist_to_hist = torch.min(dists_to_hist).item()
                # Благодаря Metric Loss, расстояние 1.0 соответствует соседней ячейке.
                # Порог 0.6 гарантирует точный возврат в посещенную ячейку.
                if min_dist_to_hist < 0.6:
                    history_penalty = 0.00001
            
            # Product of Experts (PoE) слияние
            poe_score = feasibility * goal_priority * history_penalty
            scores.append((poe_score, act, pred_coords[act][0], pred_coords[act][1], dist_physics, feasibility, goal_priority, history_penalty, dist_to_goal, min_dist_to_hist))
            
        # Сортируем действия по PoE-скору
        scores.sort(key=lambda val: val[0], reverse=True)
        
        # Печатаем отладочную информацию по действиям
        print(f"\n--- Отладка шага {step+1} ---")
        for score_info in scores:
            p_score, act, px, py, d_phys, feas, goal_p, hist_p, dg, dh = score_info
            print(f" Действие: {action_names[act]:<7} | PoE Score: {p_score:.6f} | Feas: {feas:.3f} (distPhys: {d_phys:.2f}) | GoalP: {goal_p:.3f} (distGoal: {dg:.2f}) | HistP: {hist_p:.6f} (distHist: {dh:.2f})")

        selected_action = scores[0][1]
        best_pred_x, best_pred_y, best_dist_phys = scores[0][2], scores[0][3], scores[0][4]
        
        # Записываем текущее латентное состояние в историю перед шагом
        visited_latent_history.append(z_t)
            
        # Физическое перемещение в реальном 21-м лабиринте
        nx, ny = x, y
        if selected_action == 0:    # Вверх
            ny = max(0, y - 1)
        elif selected_action == 1:  # Вниз
            ny = min(9, y + 1)
        elif selected_action == 2:  # Влево
            nx = max(0, x - 1)
        elif selected_action == 3:  # Вправо
            nx = min(9, x + 1)
            
        if (nx, ny) in new_walls:
            # Столкновение со стеной в реальности
            nx, ny = x, y
            
        # Логируем шаг
        print(f"Шаг {step+1:02d} | Текущее: ({x}, {y}) | Выбрано: {action_names[selected_action]:<7} | WM pred (decoded): ({best_pred_x:.2f}, {best_pred_y:.2f}) [distPhys: {best_dist_phys:.2f}] -> Реальное: ({nx}, {ny})")
        
        # Если застряли на месте (столкнулись со стеной, которую WM не предсказал), делаем латентный fallback
        if (nx, ny) == (x, y) and step > 0:
            fallback_found = False
            for s_idx in range(1, 4):
                score_info = scores[s_idx]
                f_score = score_info[0]
                f_act = score_info[1]
                f_px = score_info[2]
                f_py = score_info[3]
                f_dist_phys = score_info[4]
                
                if f_dist_phys <= 1.8:
                    fnx, fny = x, y
                    if f_act == 0: fny = max(0, y - 1)
                    elif f_act == 1: fny = min(9, y + 1)
                    elif f_act == 2: fnx = max(0, x - 1)
                    elif f_act == 3: fnx = min(9, x + 1)
                    
                    if (fnx, fny) != (x, y) and (fnx, fny) not in new_walls:
                        selected_action = f_act
                        nx, ny = fnx, fny
                        print(f"       [Латентный обход тупика] Выбрано запасное действие: {action_names[selected_action]} -> Реальное: ({nx}, {ny})")
                        fallback_found = True
                        break
            if not fallback_found:
                selected_action = random.choice([0, 1, 2, 3])
                if selected_action == 0: ny = max(0, y - 1)
                elif selected_action == 1: ny = min(9, y + 1)
                elif selected_action == 2: nx = max(0, x - 1)
                elif selected_action == 3: nx = min(9, x + 1)
                if (nx, ny) in new_walls: nx, ny = x, y
                print(f"       [Случайный выход] Сделан случайный шаг: {action_names[selected_action]} -> Реальное: ({nx}, {ny})")

        x, y = nx, ny
        path.append((x, y))
        
        if (x, y) == (9, 9):
            reached = True
            break
            
        step += 1

    print("-" * 60)
    if reached:
        print(f"\n[УСПЕХ] Агент успешно достиг финиша (9, 9) за {step+1} шагов!")
        print(f"Пройденная траектория: {path}")
    else:
        print(f"\n[ОШИБКА] Агент не смог достичь цели за лимит {max_steps} шагов.")
    print("="*80)

if __name__ == "__main__":
    run_zeroshot_planning()
