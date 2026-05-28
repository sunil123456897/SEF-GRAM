import torch
import torch.nn as nn
from typing import Tuple

class ARCGridEncoder(nn.Module):
    """CNN Encoder for ARC 2D Grids (30x30, 10 colors)."""
    def __init__(self, latent_dim: int, vocab_size: int = 11):
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.conv = nn.Sequential(
            nn.Conv2d(vocab_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 15x15
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3), # 5x5
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, latent_dim)
        )
        
    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        # grids: [B, T, H, W]
        B, T = grids.shape[:2]
        if grids.shape[2] != 30 or grids.shape[3] != 30:
            pad_h = 30 - grids.shape[2]
            pad_w = 30 - grids.shape[3]
            grids = torch.nn.functional.pad(grids, (0, pad_w, 0, pad_h), value=10)
        x = grids.reshape(-1, 30, 30)
        x_onehot = torch.nn.functional.one_hot(x.long(), num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        z = self.conv(x_onehot)
        return z.reshape(B, T, -1)


class ARCGridDecoder(nn.Module):
    """CNN Decoder reconstructing ARC 2D Grids from Latent State."""
    def __init__(self, latent_dim: int, vocab_size: int = 11):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 5 * 5),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, vocab_size, kernel_size=3, padding=1)
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, T, latent_dim]
        B, T, D = z.shape
        x = self.fc(z.reshape(-1, D))
        x = x.reshape(-1, 64, 5, 5)
        logits = self.deconv(x) # [B*T, vocab_size, 30, 30]
        return logits.reshape(B, T, self.vocab_size, 30, 30)


def generate_flood_fill_trajectory(B=1, T=10) -> torch.Tensor:
    """Generates synthetic ARC Flood Fill tasks.
    Background is 0, Border is 1, Fill color is 2.
    Returns: [B, T, 30, 30]
    """
    grids = torch.zeros(B, T, 30, 30, dtype=torch.long)
    for b in range(B):
        # Draw border
        top, bottom = 10, 20
        left, right = 10, 20
        grid = torch.zeros(30, 30, dtype=torch.long)
        grid[top:bottom+1, left] = 1
        grid[top:bottom+1, right] = 1
        grid[top, left:right+1] = 1
        grid[bottom, left:right+1] = 1
        
        # Seed pixel
        seed_y, seed_x = 15, 15
        grid[seed_y, seed_x] = 2
        grids[b, 0] = grid
        
        # Expand for T-1 steps
        for t in range(1, T):
            new_grid = grids[b, t-1].clone()
            y_idx, x_idx = torch.where(new_grid == 2)
            for y, x in zip(y_idx, x_idx):
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = y+dy, x+dx
                    if new_grid[ny, nx] == 0:
                        new_grid[ny, nx] = 2
            grids[b, t] = new_grid
            
    return grids


def generate_shift_trajectory(B=1, T=10) -> torch.Tensor:
    """30x30 grid. Shift a square to the right step-by-step."""
    grids = torch.zeros(B, T, 30, 30, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            grids[b, t, 10:15, 5+t:10+t] = 3
    return grids

def generate_invert_trajectory(B=1, T=10) -> torch.Tensor:
    """30x30 grid. Slowly flip colors from 0 to 4."""
    grids = torch.zeros(B, T, 30, 30, dtype=torch.long)
    for b in range(B):
        grids[b, 0, :, :15] = 4
        for t in range(1, T):
            new_grid = grids[b, t-1].clone()
            col = 15 - t
            if col >= 0:
                new_grid[:, col] = 0
            col2 = 15 + t - 1
            if col2 < 30:
                new_grid[:, col2] = 4
            grids[b, t] = new_grid
    return grids

def generate_composite_trajectory(B=1, T=10) -> torch.Tensor:
    """Composite task: Shift right and invert colors simultaneously."""
    grids = torch.zeros(B, T, 30, 30, dtype=torch.long)
    for b in range(B):
        grids[b, 0, :, :15] = 4
        for t in range(T):
            if t > 0:
                new_grid = grids[b, t-1].clone()
                col = 15 - t
                if col >= 0:
                    new_grid[:, col] = 0
                col2 = 15 + t - 1
                if col2 < 30:
                    new_grid[:, col2] = 4
            else:
                new_grid = grids[b, 0].clone()
            
            # Draw moving square
            # we need to clear previous position if not t=0
            if t > 0:
                new_grid[10:15, 5+t-1:10+t-1] = grids[b, 0, 10:15, 5+t-1:10+t-1]
            new_grid[10:15, 5+t:10+t] = 3
            grids[b, t] = new_grid
    return grids


