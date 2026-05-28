from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from sef_gram.world_model import WorldBatch, pad_obs


@dataclass
class GridWorldSpec:
    size: int = 5
    max_obs_dim: int = 16


class GridWorldBatcher:
    """Vector-state GridWorld generator.

    Observation layout before padding:
    [agent_x, agent_y, goal_x, goal_y] normalized to [0, 1].
    Actions: 0 up, 1 down, 2 left, 3 right.
    Reward: 1 on reaching the goal, otherwise -0.01.
    """

    def __init__(self, spec: GridWorldSpec):
        self.spec = spec

    def sample(self, batch_size: int, device: torch.device) -> WorldBatch:
        n = self.spec.size
        obs = []
        actions = []
        next_obs = []
        rewards = []
        dones = []
        for _ in range(batch_size):
            ax, ay = random.randrange(n), random.randrange(n)
            gx, gy = random.randrange(n), random.randrange(n)
            action = random.randrange(4)
            nx, ny = ax, ay
            if action == 0:
                ny = max(0, ay - 1)
            elif action == 1:
                ny = min(n - 1, ay + 1)
            elif action == 2:
                nx = max(0, ax - 1)
            elif action == 3:
                nx = min(n - 1, ax + 1)
            done = float(nx == gx and ny == gy)
            reward = 1.0 if done else -0.01
            obs.append([ax / (n - 1), ay / (n - 1), gx / (n - 1), gy / (n - 1)])
            next_obs.append([nx / (n - 1), ny / (n - 1), gx / (n - 1), gy / (n - 1)])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return WorldBatch(
            obs=pad_obs(torch.tensor(obs, dtype=torch.float32, device=device), self.spec.max_obs_dim),
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            next_obs=pad_obs(torch.tensor(next_obs, dtype=torch.float32, device=device), self.spec.max_obs_dim),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device),
        )


@dataclass
class LinePhysicsSpec:
    max_obs_dim: int = 16
    dt: float = 0.1
    damping: float = 0.95


class LinePhysicsBatcher:
    """Simple 1D dynamics generator.

    Observation layout before padding:
    [position, velocity, target]
    Actions: 0 accelerate left, 1 no-op, 2 accelerate right, 3 brake.
    """

    def __init__(self, spec: LinePhysicsSpec):
        self.spec = spec

    def sample(self, batch_size: int, device: torch.device) -> WorldBatch:
        obs = []
        actions = []
        next_obs = []
        rewards = []
        dones = []
        for _ in range(batch_size):
            pos = random.uniform(-1.0, 1.0)
            vel = random.uniform(-0.5, 0.5)
            target = random.uniform(-1.0, 1.0)
            action = random.randrange(4)
            accel = {0: -1.0, 1: 0.0, 2: 1.0, 3: -0.5 * vel}[action]
            next_vel = self.spec.damping * (vel + accel * self.spec.dt)
            next_pos = max(-1.0, min(1.0, pos + next_vel * self.spec.dt))
            distance = abs(next_pos - target)
            done = float(distance < 0.05)
            reward = 1.0 if done else -distance
            obs.append([pos, vel, target])
            next_obs.append([next_pos, next_vel, target])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        return WorldBatch(
            obs=pad_obs(torch.tensor(obs, dtype=torch.float32, device=device), self.spec.max_obs_dim),
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            next_obs=pad_obs(torch.tensor(next_obs, dtype=torch.float32, device=device), self.spec.max_obs_dim),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device),
        )


class MixedWorldBatcher:
    """Mixture of simple environments sharing the same WorldBatch API."""

    def __init__(self, batchers: List[object]):
        if not batchers:
            raise ValueError("MixedWorldBatcher requires at least one batcher")
        self.batchers = batchers

    def sample(self, batch_size: int, device: torch.device) -> WorldBatch:
        batcher = random.choice(self.batchers)
        return batcher.sample(batch_size, device)


def build_default_mixed_batcher(max_obs_dim: int = 16) -> MixedWorldBatcher:
    return MixedWorldBatcher(
        [
            GridWorldBatcher(GridWorldSpec(max_obs_dim=max_obs_dim)),
            LinePhysicsBatcher(LinePhysicsSpec(max_obs_dim=max_obs_dim)),
        ]
    )
