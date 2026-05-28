import math
import re
import random
import numpy as np
from collections import deque

class NQueensEnvironment:
    """
    Среда задачи N-Queens (Расстановка N ферзей).
    Используется для проверки сходимости и сквозного обучения SEF-GRAM.
    Расставляет ферзей на доске N x N так, чтобы они не били друг друга.
    """
    def __init__(self, n=8):
        self.n = n

    def parse_action_string(self, action_str):
        """
        Парсит XML-формат ответа вида: <answer>c0,c1,c2,...</answer>
        Возвращает список ходов или None в случае неверного формата.
        """
        # Ищем теги <answer>...</answer>
        pattern = r"<answer>\s*([\d,\s]+)\s*</answer>"
        match = re.search(pattern, action_str)
        if not match:
            return None
        
        try:
            # Преобразуем строку чисел через запятую в список целых чисел
            numbers = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
            return numbers
        except ValueError:
            return None

    def step(self, action):
        """
        Вычисляет три награды по целям: правильность расстановки, XML-формат, краткость.
        action: может быть строкой (с XML-тегами) или напрямую списком целых чисел.
        """
        action_sequence = None
        format_reward = 0.0
        
        if isinstance(action, str):
            # Reward Shaping: добавляем промежуточные микро-награды за теги рассуждения и ответа
            micro_reward = 0.0
            if "<think>" in action:
                micro_reward += 0.1
            if "</think>" in action:
                micro_reward += 0.1
            if "<answer>" in action:
                micro_reward += 0.1
            if "</answer>" in action:
                micro_reward += 0.1
                
            parsed = self.parse_action_string(action)
            if parsed is not None:
                action_sequence = parsed
                format_reward = 1.0  # Формат XML полностью валидный
            else:
                action_sequence = []
                # Если формат неполный, выдаем микро-награду за присутствующие теги
                format_reward = micro_reward
        else:
            # Если передан список напрямую, формат считается валидным по умолчанию
            action_sequence = list(action)
            format_reward = 1.0

        # 1. Награда за правильность (correctness)
        correctness = 1.0
        if len(action_sequence) != self.n:
            correctness = 0.0
        else:
            # Проверяем конфликты ферзей
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    col_i = action_sequence[i]
                    col_j = action_sequence[j]
                    
                    # Проверяем выход за границы доски
                    if col_i < 0 or col_i >= self.n or col_j < 0 or col_j >= self.n:
                        correctness = 0.0
                        break
                    
                    # Конфликт по одной вертикали или по диагонали
                    if col_i == col_j or abs(col_i - col_j) == abs(i - j):
                        correctness = 0.0
                        break
                if correctness == 0.0:
                    break

        # 2. Награда за краткость (brevity) - штрафуем за слишком длинный путь
        # По умолчанию из Планы.txt: exp(-0.1 * len(action_sequence))
        brevity = float(math.exp(-0.1 * len(action_sequence))) if len(action_sequence) > 0 else 0.0
        
        return {
            'correctness': correctness,
            'format': format_reward,
            'brevity': brevity
        }


class MazeGenerator:
    """
    Генератор случайных лабиринтов 10x10 с BFS-проверкой связности.
    """
    def __init__(self, size=10):
        self.size = size
        self.start = (0, 0)
        self.goal = (9, 9)

    def generate(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        
        while True:
            walls = set()
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) == self.start or (x, y) == self.goal:
                        continue
                    # 20% шанс появления стены
                    if random.random() < 0.20:
                        walls.add((x, y))
            
            # BFS проверка связности старта и финиша
            if self.has_path(walls):
                return walls

    def has_path(self, walls):
        queue = deque([self.start])
        visited = {self.start}
        
        while queue:
            x, y = queue.popleft()
            if (x, y) == self.goal:
                return True
                
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) not in walls and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return False


class SokobanGenerator:
    def __init__(self, size=5):
        self.size = size

    def generate(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        while True:
            walls = set()
            # Внешние стены
            for i in range(self.size):
                walls.add((0, i))
                walls.add((self.size - 1, i))
                walls.add((i, 0))
                walls.add((i, self.size - 1))
            
            # Внутренние препятствия (максимум 1 стена для сохранения простора)
            inner_candidates = [(x, y) for x in range(1, 4) for y in range(1, 4)]
            # Добавим случайную внутреннюю стену с вероятностью 30%
            if random.random() < 0.3:
                wx, wy = random.choice(inner_candidates)
                walls.add((wx, wy))
                
            # Свободные клетки
            free_cells = [(x, y) for x in range(1, 4) for y in range(1, 4) if (x, y) not in walls]
            if len(free_cells) < 2:
                continue
                
            random.shuffle(free_cells)
            ax, ay = free_cells[0]
            bx, by = free_cells[1]
            
            # Убедимся, что ящик не находится в углу на старте
            if not self.is_corner(bx, by, walls):
                return walls, (ax, ay), (bx, by)

    def is_corner(self, x, y, walls):
        up = (x, y - 1) in walls
        down = (x, y + 1) in walls
        left = (x - 1, y) in walls
        right = (x + 1, y) in walls
        return (up and left) or (up and right) or (down and left) or (down and right)
