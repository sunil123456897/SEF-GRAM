# Terminal Memory Capacity Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a character-level tokenization system (ASCII-260 vocabulary) and stress-testing benchmark ("Test 1: Terminal Memory Capacity Benchmark") to evaluate the capacity and OOD perplexity of `EFLACell` memory matrices.

**Architecture:** Create `CharacterTokenizer` in `sef_gram/tokenizer.py`. Implement data generators, `EFLACharacterModel` wrapper, and evaluation routines in `experiment_terminal_memory.py`. Validate tokenizer, generators, and models via `tests/test_tokenizer_and_benchmark.py`.

**Tech Stack:** PyTorch, pytest, standard Python utilities.

---

### Task 1: Create Character Tokenizer

**Files:**
- Create: `sef_gram/tokenizer.py`
- Test: `tests/test_tokenizer_and_benchmark.py`

- [ ] **Step 1: Write the failing test for CharacterTokenizer**
  Create/write to `tests/test_tokenizer_and_benchmark.py`:
  ```python
  import pytest
  from sef_gram.tokenizer import CharacterTokenizer

  def test_character_tokenizer_roundtrip():
      tokenizer = CharacterTokenizer()
      text = "Hello, World! \n\t$ 123"
      ids = tokenizer.encode(text, add_bos=True, add_eos=True)
      assert ids[0] == 1  # <bos>
      assert ids[-1] == 2 # <eos>
      decoded = tokenizer.decode(ids[1:-1])
      assert decoded == text

      # Test unknown character replacement (Unicode above 255)
      unk_text = "Hello, 世界"
      unk_ids = tokenizer.encode(unk_text)
      assert 3 in unk_ids # World chars map to unk
      assert tokenizer.decode(unk_ids).endswith("[UNK][UNK]")
  ```

- [ ] **Step 2: Run test to verify it fails**
  Run: `pytest tests/test_tokenizer_and_benchmark.py -v`
  Expected: FAIL (ModuleNotFound / Import error)

- [ ] **Step 3: Implement CharacterTokenizer**
  Create `sef_gram/tokenizer.py` with the complete implementation:
  ```python
  class CharacterTokenizer:
      """
      Посимвольный токенизатор с фиксированным ASCII-260 словарем.
      """
      def __init__(self):
          self.pad_token = "<pad>"
          self.bos_token = "<bos>"
          self.eos_token = "<eos>"
          self.unk_token = "<unk>"
          self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
          self.vocab_size = 260
          
      def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
          tokens = []
          if add_bos:
              tokens.append(1)
          for char in text:
              code = ord(char)
              if code < 256:
                  tokens.append(code + 4)
              else:
                  tokens.append(3)
          if add_eos:
              tokens.append(2)
          return tokens
          
      def decode(self, ids: list[int]) -> str:
          chars = []
          for idx in ids:
              if idx < 4:
                  if idx == 0:
                      continue  # Пропускаем паддинг
                  elif idx == 1:
                      chars.append("[BOS]")
                  elif idx == 2:
                      chars.append("[EOS]")
                  elif idx == 3:
                      chars.append("[UNK]")
              else:
                  chars.append(chr(idx - 4))
          return "".join(chars)
  ```

- [ ] **Step 4: Run test to verify it passes**
  Run: `pytest tests/test_tokenizer_and_benchmark.py::test_character_tokenizer_roundtrip -v`
  Expected: PASS

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add sef_gram/tokenizer.py tests/test_tokenizer_and_benchmark.py
  git commit -m "feat: add CharacterTokenizer and its unit tests"
  ```

---

### Task 2: Implement Benchmark Data Streams and EFLA Model

**Files:**
- Create: `experiment_terminal_memory.py`
- Modify: `tests/test_tokenizer_and_benchmark.py`

- [ ] **Step 1: Write tests for generators and character EFLA model**
  Add these tests to `tests/test_tokenizer_and_benchmark.py`:
  ```python
  import torch
  from experiment_terminal_memory import (
      EFLACharacterModel,
      SyntheticTerminalEmulator,
      HFTerminalStreamer
  )

  def test_synthetic_terminal_emulator():
      emulator = SyntheticTerminalEmulator()
      seq, target = emulator.generate_task_sequence(distractor_len=100, num_vars=2, query_var_idx=0)
      assert len(seq) > 100
      assert len(target) > 0
      assert "$" in seq
      assert "export" in seq
      assert "SECRET_PORT_0" in seq

  def test_character_model_forward():
      model = EFLACharacterModel(vocab_size=260, latent_dim=16)
      inputs = torch.randint(0, 260, (2, 20)) # batch_size=2, seq_len=20
      logits, s_next = model(inputs)
      assert logits.shape == (2, 20, 260)
      assert s_next.shape == (2, 16, 16)
  ```

- [ ] **Step 2: Run tests to verify they fail**
  Run: `pytest tests/test_tokenizer_and_benchmark.py -v`
  Expected: FAIL with ImportErrors.

- [ ] **Step 3: Implement data streams and EFLACharacterModel in experiment_terminal_memory.py**
  Create `experiment_terminal_memory.py` with data structures and the model wrapper (only data and model part for now):
  ```python
  import random
  import torch
  import torch.nn as nn
  import urllib.request
  from sef_gram.model import EFLACell
  from sef_gram.tokenizer import CharacterTokenizer

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
          self.online = True
          
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
  ```

- [ ] **Step 4: Run tests to verify they pass**
  Run: `pytest tests/test_tokenizer_and_benchmark.py -v`
  Expected: PASS all 3 tests.

- [ ] **Step 5: Commit**
  Run:
  ```bash
  git add experiment_terminal_memory.py tests/test_tokenizer_and_benchmark.py
  git commit -m "feat: implement EFLACharacterModel, SyntheticTerminalEmulator, HFTerminalStreamer"
  ```

---

### Task 3: Implement Training and Capacity Evaluation Suite

**Files:**
- Modify: `experiment_terminal_memory.py`

- [ ] **Step 1: Append training and evaluation loop to experiment_terminal_memory.py**
  Write the training loop, loss calculation, evaluation loops, and main entry point using `MuonWithAuxAdam` (from `sef_gram.optimization`):
  ```python
  import torch.nn.functional as F
  from sef_gram.optimization import MuonWithAuxAdam

  def train_character_model(model, train_sequences, val_sequences, epochs=10):
      """
      Обучает посимвольную EFLA-модель с использованием MuonWithAuxAdam.
      """
      tokenizer = CharacterTokenizer()
      optimizer = MuonWithAuxAdam(model.parameters(), lr=0.02, adamw_lr=3e-4)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
      
      print(f"Starting SFT Slices on Device: {device}...")
      
      for epoch in range(epochs):
          model.train()
          total_loss = 0
          for seq in train_sequences:
              optimizer.zero_grad()
              
              ids = torch.tensor(tokenizer.encode(seq, add_bos=True, add_eos=True), dtype=torch.long, device=device).unsqueeze(0)
              if ids.shape[1] < 2:
                  continue
                  
              inputs = ids[:, :-1]
              targets = ids[:, 1:]
              
              logits, _ = model(inputs)
              loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
              loss.backward()
              
              # Градиентный клиппинг
              torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
              
              optimizer.step()
              total_loss += loss.item()
              
          model.eval()
          val_loss = 0
          with torch.no_grad():
              for seq in val_sequences:
                  ids = torch.tensor(tokenizer.encode(seq, add_bos=True, add_eos=True), dtype=torch.long, device=device).unsqueeze(0)
                  if ids.shape[1] < 2:
                      continue
                  inputs = ids[:, :-1]
                  targets = ids[:, 1:]
                  logits, _ = model(inputs)
                  loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                  val_loss += loss.item()
                  
          print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss/max(1, len(train_sequences)):.4f} (nats) | Val Loss: {val_loss/max(1, len(val_sequences)):.4f} (nats)")
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
      train_data = [streamer.get_random_task() for _ in range(50)]
      val_data = [streamer.get_random_task() for _ in range(10)]
      
      print("Streamer Mode online:", streamer.online)
      
      # 2. Запускаем обучение и тестирование для различных размерностей S_t
      for d in [16, 32]:
          print(f"\n==========================================")
          print(f"Training Model with EFLACell dimension {d}x{d}")
          print(f"==========================================")
          model = EFLACharacterModel(vocab_size=260, latent_dim=d)
          
          # Обучаем модель
          model = train_character_model(model, train_data, val_data, epochs=5)
          
          # Оцениваем емкость памяти
          evaluate_retrieval_capacity(model, latent_dim=d)
  ```

- [ ] **Step 2: Verify the whole suite runs synchronously**
  Propose running the script with the built-in system python interpreter to verify stdout logs.
  Run command: `python experiment_terminal_memory.py`
  Expected: Execution finishes cleanly with loss output and capacity matrix sweeps printed.

- [ ] **Step 3: Commit**
  Run:
  ```bash
  git add experiment_terminal_memory.py
  git commit -m "feat: complete SFT training loop and capacity sweeps in experiment_terminal_memory"
  ```
