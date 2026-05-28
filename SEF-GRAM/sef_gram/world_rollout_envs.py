from __future__ import annotations

import torch

from sef_gram.world_model import WorldBatch, pad_obs


class GridWorldRolloutEnv:
    """Stateful/vectorized GridWorld transition model for true rollout evaluation."""

    obs_dim = 4
    num_actions = 4

    def __init__(self, size: int = 5, max_obs_dim: int = 16):
        self.size = size
        self.max_obs_dim = max_obs_dim

    def reset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n = self.size
        agent = torch.randint(0, n, (batch_size, 2), device=device).float()
        goal = torch.randint(0, n, (batch_size, 2), device=device).float()
        obs = torch.cat([agent, goal], dim=-1) / float(n - 1)
        return pad_obs(obs, self.max_obs_dim)

    def sample_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_actions, (batch_size,), device=device)

    def step_from_obs(self, obs: torch.Tensor, actions: torch.Tensor) -> WorldBatch:
        n = self.size
        raw = obs[:, : self.obs_dim].clamp(0.0, 1.0)
        scale = float(n - 1)
        ax = torch.round(raw[:, 0] * scale).long().clamp(0, n - 1)
        ay = torch.round(raw[:, 1] * scale).long().clamp(0, n - 1)
        gx = torch.round(raw[:, 2] * scale).long().clamp(0, n - 1)
        gy = torch.round(raw[:, 3] * scale).long().clamp(0, n - 1)

        nx = ax.clone()
        ny = ay.clone()
        actions = actions.long().view(-1)
        ny = torch.where(actions == 0, (ny - 1).clamp(0, n - 1), ny)
        ny = torch.where(actions == 1, (ny + 1).clamp(0, n - 1), ny)
        nx = torch.where(actions == 2, (nx - 1).clamp(0, n - 1), nx)
        nx = torch.where(actions == 3, (nx + 1).clamp(0, n - 1), nx)

        done = ((nx == gx) & (ny == gy)).float()
        reward = torch.where(done > 0, torch.ones_like(done), torch.full_like(done, -0.01))
        next_raw = torch.stack([nx.float(), ny.float(), gx.float(), gy.float()], dim=-1) / scale
        next_obs = pad_obs(next_raw, self.max_obs_dim)
        return WorldBatch(obs=obs, actions=actions, next_obs=next_obs, rewards=reward, dones=done)


class KeyDoorGridWorldRolloutEnv:
    """Stateful/vectorized key-door gridworld for causal rollout evaluation."""

    obs_dim = 9
    num_actions = 4

    def __init__(self, size: int = 6, max_obs_dim: int = 16):
        self.size = size
        self.max_obs_dim = max_obs_dim

    def reset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        n = self.size
        agent = torch.randint(0, n, (batch_size, 2), device=device).float()
        key = torch.randint(0, n, (batch_size, 2), device=device).float()
        door = torch.randint(0, n, (batch_size, 2), device=device).float()
        goal = torch.randint(0, n, (batch_size, 2), device=device).float()
        has_key = torch.randint(0, 2, (batch_size, 1), device=device).float()
        obs = torch.cat([agent, key, door, goal], dim=-1) / float(n - 1)
        obs = torch.cat([obs, has_key], dim=-1)
        return pad_obs(obs, self.max_obs_dim)

    def sample_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_actions, (batch_size,), device=device)

    def step_from_obs(self, obs: torch.Tensor, actions: torch.Tensor) -> WorldBatch:
        n = self.size
        raw = obs[:, : self.obs_dim].clone()
        scale = float(n - 1)
        pos = torch.round(raw[:, :8].clamp(0.0, 1.0) * scale).long().clamp(0, n - 1)
        ax, ay = pos[:, 0], pos[:, 1]
        kx, ky = pos[:, 2], pos[:, 3]
        dx, dy = pos[:, 4], pos[:, 5]
        gx, gy = pos[:, 6], pos[:, 7]
        has_key = (raw[:, 8] > 0.5).long()

        nx = ax.clone()
        ny = ay.clone()
        actions = actions.long().view(-1)
        ny = torch.where(actions == 0, (ny - 1).clamp(0, n - 1), ny)
        ny = torch.where(actions == 1, (ny + 1).clamp(0, n - 1), ny)
        nx = torch.where(actions == 2, (nx - 1).clamp(0, n - 1), nx)
        nx = torch.where(actions == 3, (nx + 1).clamp(0, n - 1), nx)

        blocked = (nx == dx) & (ny == dy) & (has_key == 0)
        nx = torch.where(blocked, ax, nx)
        ny = torch.where(blocked, ay, ny)

        collected_key = (nx == kx) & (ny == ky)
        next_has_key = ((has_key == 1) | collected_key).long()
        done = ((next_has_key == 1) & (nx == gx) & (ny == gy)).float()
        reward = torch.full_like(done, -0.02)
        reward = torch.where(blocked, torch.full_like(reward, -0.1), reward)
        reward = torch.where((has_key == 0) & (next_has_key == 1), torch.full_like(reward, 0.2), reward)
        reward = torch.where(done > 0, torch.ones_like(reward), reward)

        next_raw = torch.stack(
            [nx.float(), ny.float(), kx.float(), ky.float(), dx.float(), dy.float(), gx.float(), gy.float()], dim=-1
        ) / scale
        next_raw = torch.cat([next_raw, next_has_key.float().unsqueeze(-1)], dim=-1)
        next_obs = pad_obs(next_raw, self.max_obs_dim)
        return WorldBatch(obs=obs, actions=actions, next_obs=next_obs, rewards=reward, dones=done)


class LinePhysicsRolloutEnv:
    """Stateful/vectorized 1D physics transition model for true rollout evaluation."""

    obs_dim = 3
    num_actions = 4

    def __init__(self, max_obs_dim: int = 16, dt: float = 0.1, damping: float = 0.95):
        self.max_obs_dim = max_obs_dim
        self.dt = dt
        self.damping = damping

    def reset(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pos = torch.empty(batch_size, device=device).uniform_(-1.0, 1.0)
        vel = torch.empty(batch_size, device=device).uniform_(-0.5, 0.5)
        target = torch.empty(batch_size, device=device).uniform_(-1.0, 1.0)
        return pad_obs(torch.stack([pos, vel, target], dim=-1), self.max_obs_dim)

    def sample_actions(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_actions, (batch_size,), device=device)

    def step_from_obs(self, obs: torch.Tensor, actions: torch.Tensor) -> WorldBatch:
        raw = obs[:, : self.obs_dim]
        pos = raw[:, 0].clamp(-1.0, 1.0)
        vel = raw[:, 1].clamp(-2.0, 2.0)
        target = raw[:, 2].clamp(-1.0, 1.0)
        actions = actions.long().view(-1)

        accel = torch.zeros_like(pos)
        accel = torch.where(actions == 0, torch.full_like(accel, -1.0), accel)
        accel = torch.where(actions == 2, torch.full_like(accel, 1.0), accel)
        accel = torch.where(actions == 3, -0.5 * vel, accel)

        next_vel = self.damping * (vel + accel * self.dt)
        next_pos = (pos + next_vel * self.dt).clamp(-1.0, 1.0)
        distance = (next_pos - target).abs()
        done = (distance < 0.05).float()
        reward = torch.where(done > 0, torch.ones_like(done), -distance)
        next_obs = pad_obs(torch.stack([next_pos, next_vel, target], dim=-1), self.max_obs_dim)
        return WorldBatch(obs=obs, actions=actions, next_obs=next_obs, rewards=reward, dones=done)


def rollout_envs(max_obs_dim: int = 16):
    return {
        "gridworld": GridWorldRolloutEnv(max_obs_dim=max_obs_dim),
        "key_door_gridworld": KeyDoorGridWorldRolloutEnv(max_obs_dim=max_obs_dim),
        "line_physics": LinePhysicsRolloutEnv(max_obs_dim=max_obs_dim),
    }
