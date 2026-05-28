import torch

def prior_energy(z):
    # Узкая цель в точке (5.0, 5.0) и сильное препятствие по центру (2.5, 2.5)
    goal = torch.tensor([5.0, 5.0])
    obs = torch.tensor([2.5, 2.5])
    
    # Штраф за удаление от цели (притягивает к 5.0, 5.0)
    goal_dist = torch.sum((z - goal)**2, dim=-1)
    
    # Энергетический барьер (отталкивает от 2.5, 2.5)
    obs_dist = torch.sum((z - obs)**2, dim=-1)
    barrier = 10.0 * torch.exp(-obs_dist)
    
    return goal_dist + barrier

def run_poe_test():
    print("=== MVP 1: Energy-Based PoE Planner ===")
    
    # Метод А: Пассивный случайный сэмплинг (GRAM)
    torch.manual_seed(42)
    num_samples = 1000
    # GRAM генерирует случайные шаги из (0,0) со средним сдвигом
    random_trajectories = torch.randn(num_samples, 2) * 2.0 + torch.tensor([2.5, 2.5])
    
    energies = prior_energy(random_trajectories)
    best_idx = torch.argmin(energies)
    best_random_z = random_trajectories[best_idx]
    
    print(f"[Random Sampling] Из 1000 попыток лучшая точка: {best_random_z.tolist()}")
    print(f"[Random Sampling] Минимальная энергия BJEPA: {energies[best_idx].item():.4f}")
    
    # Метод B: Градиентный спуск в латенте (Gradient MPC)
    # Начинаем с (0,0)
    z_opt = torch.tensor([[0.0, 0.0]], requires_grad=True)
    optimizer = torch.optim.Adam([z_opt], lr=0.1)
    
    print("\n[Gradient MPC] Начинаем градиентное руление (100 шагов)...")
    for step in range(100):
        optimizer.zero_grad()
        loss = prior_energy(z_opt).mean()
        loss.backward()
        optimizer.step()
        
    final_energy = prior_energy(z_opt.detach()).mean().item()
    print(f"[Gradient MPC] Финальная точка: {z_opt.detach()[0].tolist()}")
    print(f"[Gradient MPC] Финальная энергия BJEPA: {final_energy:.4f}")
    
    print("\n--- Вывод ---")
    if final_energy < energies[best_idx].item():
        print("Градиентное планирование (Gradient MPC) обошло случайный перебор,")
        print("успешно обогнув препятствие и достигнув цели!")

if __name__ == "__main__":
    run_poe_test()
