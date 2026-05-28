import math

def run_dirichlet_lambda_test():
    print("=== MVP 4: Dynamic Dirichlet Lambda ===")
    
    print("Эмуляция 5 эпох обучения. Энергия Дирихле (DE) падает по мере того,")
    print("как модель выучивает изометрию пространства (Ага!-момент).")
    
    print("\nЭпоха | Dirichlet Energy | Env Lambda (ECHO) | GDPO Weight | Вывод")
    print("-" * 75)
    
    base_gdpo_weight = 1.0
    
    for epoch in range(1, 6):
        # Имитация падения энергии: от 10.0 (хаос) до 0.1 (порядок)
        dirichlet_energy = 10.0 * math.exp(-1.2 * (epoch - 1))
        
        # Динамическая лямбда: если энергия > 2.0, мы еще в фазе зубрежки
        # Полностью концентрируемся на ECHO (предсказание среды)
        # Если энергия упала < 2.0, плавно снижаем лямбду, перенося фокус на RL (GDPO)
        
        env_lambda = min(1.0, dirichlet_energy / 2.0)
        
        # Чем меньше env_lambda, тем больше доминирует GDPO
        conclusion = ""
        if env_lambda > 0.8:
            conclusion = "Зубрежка логов (ECHO)"
        elif env_lambda < 0.2:
            conclusion = "RL-Выравнивание (GDPO)"
        else:
            conclusion = "Ага!-момент (Переход)"
            
        print(f"  {epoch}   |       {dirichlet_energy:5.2f}      |       {env_lambda:.2f}        |     {base_gdpo_weight:.2f}    | {conclusion}")
        
    print("\n--- Вывод ---")
    print("Динамическая лямбда позволяет агенту сначала выучить физику мира (ECHO),")
    print("а сразу после наступления Ага!-момента (падение DE) переключиться на")
    print("поиск решения задачи, не тратя эпохи впустую!")

if __name__ == "__main__":
    run_dirichlet_lambda_test()
