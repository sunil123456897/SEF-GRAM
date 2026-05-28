import torch
import torch.nn.functional as F

def run_discrete_attractor_test():
    print("=== MVP 3: EFLA Discrete Attractor ===")
    
    # 2 Дискретных токена (например, '<' и '>')
    # Их эмбеддинги в латентном пространстве
    vocab_embeddings = torch.tensor([[-1.0], [1.0]])
    
    # Непрерывная динамика EFLA смещает вектор от -1.0 к 1.0 (дрейф)
    # Предположим, идеальный шаг должен был перекинуть нас ровно в 1.0
    # Но из-за гладкости ODE мы не долетели и оказались в 0.2
    raw_z_t = torch.tensor([[0.2]], requires_grad=True)
    
    print(f"Сырое состояние после гладкой динамики: {raw_z_t.item():.4f}")
    
    # Без аттрактора (Softmax над расстояниями)
    dists = torch.cdist(raw_z_t, vocab_embeddings)
    probs = F.softmax(-dists * 2.0, dim=-1) # * 2.0 это temperature
    print(f"Вероятности токенов БЕЗ регуляризации: Токен 0: {probs[0][0].item():.4f}, Токен 1: {probs[0][1].item():.4f}")
    print("-> Ошибка! Модель не уверена, синтаксис может сломаться.")
    
    # С аттрактором (Оптимизация скрытого состояния)
    optimizer = torch.optim.SGD([raw_z_t], lr=0.5)
    
    # Делаем пару шагов, притягивая сырое состояние к БЛИЖАЙШЕМУ токену
    # (эмуляция L2 Attractor Loss во время обучения)
    for _ in range(5):
        optimizer.zero_grad()
        # Ищем ближайший (в данном случае Токен 1 (1.0))
        dists = torch.cdist(raw_z_t, vocab_embeddings)
        nearest_idx = torch.argmin(dists, dim=-1)
        nearest_emb = vocab_embeddings[nearest_idx]
        
        # Loss - это расстояние до ближайшего "валидного" токена
        loss = F.mse_loss(raw_z_t, nearest_emb)
        loss.backward()
        optimizer.step()
        
    print(f"\nСостояние после действия магнитного аттрактора: {raw_z_t.item():.4f}")
    
    dists_new = torch.cdist(raw_z_t, vocab_embeddings)
    probs_new = F.softmax(-dists_new * 2.0, dim=-1)
    print(f"Вероятности токенов С регуляризацией: Токен 0: {probs_new[0][0].item():.4f}, Токен 1: {probs_new[0][1].item():.4f}")
    
    print("\n--- Вывод ---")
    if probs_new[0][1].item() > 0.9:
        print("Непрерывный вектор 'засосало' в дискретный токен.")
        print("Аттракторы защитят EFLA от 'дискретного сбоя' при генерации кода!")

if __name__ == "__main__":
    run_discrete_attractor_test()
