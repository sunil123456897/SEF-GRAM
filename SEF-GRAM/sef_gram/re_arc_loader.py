import zipfile
import json
import torch
import random
import os

class ReArcDataset:
    def __init__(self, zip_path="E:/experiments/SEF-GRAM/SEF-GRAM/re_arc/re_arc.zip"):
        self.zip_path = zip_path
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"RE-ARC zip not found at {self.zip_path}. Please check git clone.")
            
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            self.task_files = [f for f in z.namelist() if f.endswith('.json') and 'tasks/' in f and '__MACOSX' not in f]
            
    def _pad_grid(self, grid, target_size=30, pad_val=10):
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        padded = [[pad_val]*target_size for _ in range(target_size)]
        for i in range(min(h, target_size)):
            for j in range(min(w, target_size)):
                padded[i][j] = grid[i][j]
        return padded

    def get_batch(self, batch_size=4):
        tasks = random.sample(self.task_files, batch_size)
        
        support_obs_list = []
        support_next_list = []
        query_obs_list = []
        query_next_list = []
        
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for task_file in tasks:
                with z.open(task_file) as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    train_exs = data.get('train', [])[:3]
                    test_exs = data.get('test', [])[:1]
                else:
                    train_exs = data[:3]
                    test_exs = data[-1:]
                
                while len(train_exs) < 3:
                    train_exs.append(train_exs[-1] if train_exs else {'input':[[0]], 'output':[[0]]})
                if not test_exs:
                    test_exs = [{'input':[[0]], 'output':[[0]]}]
                    
                # Augmentations per task
                color_perm = list(range(10))
                random.shuffle(color_perm)
                color_perm.append(10) # PAD is 10
                
                k_rot = random.randint(0, 3)
                flip = random.choice([True, False])
                
                def apply_aug(grid):
                    padded = self._pad_grid(grid)
                    padded = [[color_perm[c] for c in row] for row in padded]
                    if flip:
                        padded = [row[::-1] for row in padded]
                    for _ in range(k_rot):
                        padded = [list(x) for x in zip(*padded[::-1])]
                    return padded
                    
                s_obs = []
                s_next = []
                for ex in train_exs:
                    s_obs.append(apply_aug(ex['input']))
                    s_next.append(apply_aug(ex['output']))
                    
                support_obs_list.append(s_obs)
                support_next_list.append(s_next)
                
                query_obs_list.append([apply_aug(test_exs[0]['input'])])
                query_next_list.append([apply_aug(test_exs[0]['output'])])
                
        so = torch.tensor(support_obs_list, dtype=torch.long)
        sn = torch.tensor(support_next_list, dtype=torch.long)
        qo = torch.tensor(query_obs_list, dtype=torch.long)
        qn = torch.tensor(query_next_list, dtype=torch.long)
        
        return so, sn, qo, qn
