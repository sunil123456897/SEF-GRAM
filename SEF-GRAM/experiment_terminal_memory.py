import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
from sef_gram.model import EFLACell
from sef_gram.tokenizer import CharacterTokenizer
from sef_gram.optimization import MuonWithAuxAdam

class EFLACharacterModel(nn.Module):
    """
    Посимвольная языковая модель на базе EFLACell.
    """
    def __init__(self, vocab_size=260, latent_dim=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(vocab_size, latent_dim)
        self.cell = EFLACell(latent_dim=latent_dim)
        self.head = nn.Linear(latent_dim, vocab_size)

    def forward(self, input_ids, s_prev=None):
        B, T = input_ids.shape
        device = input_ids.device
        
        if s_prev is None:
            s_prev = torch.zeros(B, self.latent_dim, self.latent_dim, device=device)
            
        embeddings = self.embedding(input_ids) # (B, T, D)
        
        logits = []
        s_curr = s_prev
        
        for t in range(T):
            e_t = embeddings[:, t, :] # (B, D)
            z_curr, s_curr = self.cell(e_t, s_curr)
            out = self.head(z_curr) # (B, vocab_size)
            logits.append(out.unsqueeze(1))
            
        logits = torch.cat(logits, dim=1) # (B, T, vocab_size)
        return logits, s_curr


class SyntheticTerminalEmulator:
    """
    Генератор синтетических сессий для стресс-тестирования памяти.
    """
    def __init__(self):
        self.log_templates = [
            "INFO: [{time}] System startup sequence initialized.",
            "WARNING: [{time}] CPU usage exceeded threshold: {cpu}%.",
            "ERROR: [{time}] Database connection timeout on port {port}.",
            "DEBUG: [{time}] Memory heap compaction complete in {ms}ms.",
            "INFO: [{time}] Incoming connection accepted from {ip}:{port}.",
            "INFO: [{time}] User {username} logged in successfully.",
            "WARNING: [{time}] Slow response detected on GET /api/v1/resources.",
        ]
        
    def generate_distractor(self, length: int) -> str:
        text_parts = []
        current_len = 0
        while current_len < length:
            template = random.choice(self.log_templates)
            log_str = template.format(
                time=f"2026-05-28 {random.randint(10,12)}:{random.randint(10,59)}:{random.randint(10,59)}",
                cpu=random.randint(80, 99),
                port=random.choice([5432, 3306, 6379, 8080]),
                ms=random.randint(10, 250),
                ip=f"192.168.1.{random.randint(2, 254)}",
                username=random.choice(["admin", "vanya", "guest", "root"])
            ) + "\n"
            text_parts.append(log_str)
            current_len += len(log_str)
        return "".join(text_parts)[:length]

    def generate_task_sequence(self, distractor_len: int, num_vars: int = 1, query_var_idx: int = 0) -> tuple[str, str]:
        variables = []
        for i in range(num_vars):
            var_name = f"SECRET_PORT_{i}"
            var_val = str(random.randint(1000, 9999))
            variables.append((var_name, var_val))
            
        store_str = ""
        for var_name, var_val in variables:
            store_str += f"$ export {var_name}={var_val}\n"
            store_str += f"> Variable {var_name} defined successfully.\n"
            
        dist_str = "$ cat system_logs.log\n" + self.generate_distractor(distractor_len) + "\n"
        
        query_var_name, query_var_val = variables[query_var_idx]
        query_str = f"$ echo ${query_var_name}\n> "
        
        full_seq = store_str + dist_str + query_str
        target = query_var_val + "\n"
        return full_seq, target


class HFTerminalStreamer:
    """
    Стример логов из датасета Endless Terminals с оффлайн-фоллбэком.
    """
    def __init__(self):
        # Отключаем онлайн-запросы по умолчанию для мгновенного выполнения без сетевых задержек
        self.online = False
        
    def generate_fallback_session(self) -> str:
        sessions = [
            "$ git init\nInitialized empty Git repository in /home/user/app/.git/\n"
            "$ git status\nOn branch main\nNo commits yet\nUntracked files:\n  requirements.txt\n  main.py\n"
            "$ git add .\n$ git commit -m \"initial commit\"\n[main (root-commit) a1b2c3d] initial commit\n 2 files changed, 45 insertions(+)\n",
            "$ docker build -t test-app .\nSending build context to Docker daemon  4.5kB\nStep 1/3 : FROM python:3.9-slim\n ---> 1e92d04a601c\nStep 2/3 : COPY . /app\n ---> Using cache\nStep 3/3 : CMD [\"python\", \"/app/main.py\"]\n ---> Using cache\nSuccessfully built abc123def456\nSuccessfully tagged test-app:latest\n",
            "$ python main.py\nTraceback (most recent call last):\n  File \"main.py\", line 12, in <module>\n    run_server()\n  File \"main.py\", line 8, in run_server\n    port = int(os.environ[\"PORT\"])\n  File \"/usr/lib/python3.9/os.py\", line 679, in __getitem__\n    raise KeyError(key) from None\nKeyError: 'PORT'\n",
            "$ for i in {1..3}; do echo \"Processing chunk $i\"; done\nProcessing chunk 1\nProcessing chunk 2\nProcessing chunk 3\n$ grep \"ERROR\" server.log\nERROR: [2026-05-28 11:13:00] Database unreachable.\n"
        ]
        return random.choice(sessions)
        
    def get_random_task(self) -> str:
        task_hashes = ["0033979a", "003d339f", "0063591c", "0090c771", "009a1afa", "00ac916d", "00b7d96d"]
        selected_hash = random.choice(task_hashes)
        url = f"https://huggingface.co/datasets/obiwan96/endless-terminals/raw/main/task_000000_{selected_hash}/solution/solve.sh"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=2) as response:
                content = response.read().decode('utf-8')
                return f"$ ./solve.sh\n{content}\n"
        except Exception:
            self.online = False
            return self.generate_fallback_session()


def train_character_model(model, train_sequences, val_sequences, epochs=5):
    """
    Обучает посимвольную EFLA-модель с использованием MuonWithAuxAdam.
    Все последовательности обрабатываются параллельно в одном батче с паддингом.
    """
    tokenizer = CharacterTokenizer()
    optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Starting SFT Slices on Device: {device}...")
    
    # Функция для паддинга батча
    def collate_sequences(sequences):
        encoded = [tokenizer.encode(seq, add_bos=True, add_eos=True) for seq in sequences]
        max_len = max(len(ids) for ids in encoded)
        padded = []
        for ids in encoded:
            padded.append(ids + [0] * (max_len - len(ids)))
        return torch.tensor(padded, dtype=torch.long, device=device)

    # Собираем данные в единые параллельные тензоры
    train_tensor = collate_sequences(train_sequences)
    val_tensor = collate_sequences(val_sequences)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        inputs = train_tensor[:, :-1]
        targets = train_tensor[:, 1:]
        
        logits, _ = model(inputs)
        # ignore_index=0 игнорирует спецтокен паддинга при вычислении лосса
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            v_inputs = val_tensor[:, :-1]
            v_targets = val_tensor[:, 1:]
            v_logits, _ = model(v_inputs)
            val_loss = F.cross_entropy(v_logits.reshape(-1, v_logits.size(-1)), v_targets.reshape(-1), ignore_index=0)
            
        print(f"Epoch {epoch+1:02d} | Train Loss: {loss.item():.4f} (nats) | Val Loss: {val_loss.item():.4f} (nats)")
    return model


def evaluate_retrieval_capacity(model, latent_dim, sweeps=[50, 100, 250, 500, 1000]):
    """
    Проводит стресс-тест памяти: оценивает точность ассоциативного вызова
    при различной длине дистрактора L.
    """
    tokenizer = CharacterTokenizer()
    emulator = SyntheticTerminalEmulator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"\n--- CAPACITY EVALUATION REPORT (Matrix S_t size: {latent_dim}x{latent_dim}) ---")
    print(f"{'Distractor (chars)':<20} | {'Accuracy (%)':<15} | {'Average Nats/Token':<20}")
    print("-" * 65)
    
    for L in sweeps:
        correct = 0
        total_loss = 0
        test_cases = 20
        
        for _ in range(test_cases):
            full_seq, target_val = emulator.generate_task_sequence(distractor_len=L, num_vars=1, query_var_idx=0)
            
            # Кодируем входную последовательность и цель
            ids = torch.tensor(tokenizer.encode(full_seq, add_bos=True), dtype=torch.long, device=device).unsqueeze(0)
            target_ids = tokenizer.encode(target_val)
            
            with torch.no_grad():
                # 1. Пропускаем всю входную последовательность
                logits, s_state = model(ids)
                
                # 2. Генерируем ответ символ за символом
                generated_ids = []
                curr_in = ids[:, -1:]
                
                for char_idx in range(len(target_ids)):
                    logits_step, s_state = model(curr_in, s_prev=s_state)
                    next_id = torch.argmax(logits_step[0, -1, :]).item()
                    generated_ids.append(next_id)
                    curr_in = torch.tensor([[next_id]], dtype=torch.long, device=device)
                    
                gen_text = tokenizer.decode(generated_ids)
                
                if gen_text.strip() == target_val.strip():
                    correct += 1
                    
                # Расчет nats/token
                target_t = torch.tensor(target_ids, dtype=torch.long, device=device)
                inputs_t = torch.tensor([ids[0, -1].item()] + target_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                logits_target, _ = model(inputs_t, s_prev=s_state)
                loss = F.cross_entropy(logits_target.view(-1, logits_target.size(-1)), target_t.view(-1))
                total_loss += loss.item()
                
        acc = (correct / test_cases) * 100
        avg_loss = total_loss / test_cases
        print(f"{L:<20} | {acc:<15.1f} | {avg_loss:<20.4f}")


if __name__ == "__main__":
    # 1. Создаем стример и собираем тренировочные данные
    streamer = HFTerminalStreamer()
    emulator = SyntheticTerminalEmulator()
    print("Collecting dataset traces...")
    
    # Mixed-SFT: 35 обычных логов + 15 синтетических цепочек извлечения переменных (30% доли)
    train_data = [streamer.get_random_task() for _ in range(35)]
    for _ in range(15):
        seq, target = emulator.generate_task_sequence(distractor_len=random.randint(50, 150))
        train_data.append(seq + target)
        
    val_data = [streamer.get_random_task() for _ in range(7)]
    for _ in range(3):
        seq, target = emulator.generate_task_sequence(distractor_len=random.randint(50, 150))
        val_data.append(seq + target)
    
    print("Streamer Mode online:", streamer.online)
    
    # 2. Запускаем обучение и тестирование для различных размерностей S_t
    for d in [16, 32]:
        print(f"\n==========================================")
        print(f"Training Model with EFLACell dimension {d}x{d}")
        print(f"==========================================")
        model = EFLACharacterModel(vocab_size=260, latent_dim=d)
        
        # Обучаем модель на 25 эпох для полной корректировки смещения
        model = train_character_model(model, train_data, val_data, epochs=25)
        
        # Оцениваем емкость памяти
        evaluate_retrieval_capacity(model, latent_dim=d)
