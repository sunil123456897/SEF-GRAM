from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

Tensor = torch.Tensor


@dataclass
class TurnBatch:
    obs: Tensor
    env_target: Tensor
    reward: Tensor
    done: Tensor
    valid_actions: Tensor


class MiniTerminalEnv:
    """Synthetic terminal filesystem (Phase 2 compatible). See original docs."""

    NUM_ACTIONS = 11
    NUM_ENV_CLASSES = 6
    OBS_DIM = 15
    MAX_TURNS = 6

    def __init__(self, num_files: int = 5, num_words: int = 5, seed: int = 42):
        self.num_files = num_files
        self.num_words = num_words
        self._seed = seed
        self.rng = None

    def _random_state(self, shape: Tuple[int, ...], device: torch.device) -> Tuple[Tensor, Tensor]:
        if self.rng is None or self.rng.device != device:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(self._seed)
        contents = torch.randint(0, self.num_words, (*shape, self.num_files), device=device, generator=self.rng)
        target = torch.randint(0, self.num_words, shape, device=device, generator=self.rng)
        target_in = (contents == target.unsqueeze(-1)).any(dim=-1)
        needs_fix = ~target_in
        if needs_fix.any():
            fix_idx = torch.randint(0, self.num_files, shape, device=device, generator=self.rng)
            flat_needs = needs_fix.reshape(-1)
            flat_contents = contents.reshape(-1, self.num_files)
            flat_fix = fix_idx.reshape(-1)
            flat_target = target.reshape(-1)
            for i in range(flat_needs.shape[0]):
                if flat_needs[i]:
                    flat_contents[i, flat_fix[i]] = flat_target[i]
            contents = flat_contents.reshape(*shape, self.num_files)
        return contents, target

    def _encode_obs(self, visible: Tensor, revealed: Tensor, revealed_content: Tensor, target: Tensor) -> Tensor:
        obs_shape = visible.shape[:-1]
        visible_f = visible.float().reshape(-1, self.num_files)
        read_f = revealed.float().reshape(-1, self.num_files)
        target_oh = torch.zeros(visible_f.shape[0], self.num_words, device=visible.device)
        target_flat = target.reshape(-1, 1)
        target_oh.scatter_(1, target_flat, 1.0)
        obs = torch.cat([visible_f, read_f, target_oh], dim=-1)
        return obs.reshape(*obs_shape, self.OBS_DIM)

    def reset(self, batch_size: int, device: torch.device, num_candidates: int = 1) -> TurnBatch:
        shape = (batch_size, num_candidates) if num_candidates > 1 else (batch_size,)
        contents, target = self._random_state(shape, device)
        visible = torch.zeros(*shape, self.num_files, dtype=torch.bool, device=device)
        revealed = torch.zeros(*shape, self.num_files, dtype=torch.bool, device=device)
        revealed_content = torch.full((*shape, self.num_files), -1, dtype=torch.long, device=device)
        self._state_shape = shape
        self._contents = contents
        self._target = target
        self._visible = visible
        self._revealed = revealed
        self._revealed_content = revealed_content
        obs = self._encode_obs(visible, revealed, revealed_content, target)
        valid = torch.ones(*shape, self.NUM_ACTIONS, dtype=torch.bool, device=device)
        valid[..., 0] = ~visible.any(dim=-1)
        for i in range(self.num_files):
            valid[..., i + 1] = visible[..., i] & ~revealed[..., i]
            valid[..., i + 6] = visible[..., i]
        return TurnBatch(obs=obs, env_target=torch.zeros(*shape, dtype=torch.long, device=device), reward=torch.zeros(*shape, device=device), done=torch.zeros(*shape, dtype=torch.bool, device=device), valid_actions=valid)

    def step(self, actions: Tensor) -> TurnBatch:
        shape = self._state_shape
        ls_mask = actions == 0
        read_mask = (actions >= 1) & (actions <= self.num_files)
        answer_mask = (actions >= self.num_files + 1) & (actions <= self.num_files * 2)
        env_target = torch.zeros(*shape, dtype=torch.long, device=actions.device)
        reward = torch.zeros(*shape, device=actions.device)
        done = torch.zeros(*shape, dtype=torch.bool, device=actions.device)
        if ls_mask.any():
            self._visible[ls_mask] = True
        if read_mask.any():
            file_idx = (actions[read_mask] - 1).long()
            idx_tuple = torch.where(read_mask)
            contents_idx = self._contents[idx_tuple]
            content = contents_idx[torch.arange(len(idx_tuple[0])), file_idx]
            reveal_idx = (*idx_tuple, file_idx)
            self._revealed[reveal_idx] = True
            self._revealed_content[reveal_idx] = content
            env_target[read_mask] = content + 1
        if answer_mask.any():
            file_idx = (actions[answer_mask] - self.num_files - 1).long()
            idx_tuple = torch.where(answer_mask)
            contents_idx = self._contents[idx_tuple]
            content = contents_idx[torch.arange(len(idx_tuple[0])), file_idx]
            target_idx = self._target[idx_tuple]
            correct = (content == target_idx).float()
            reward[answer_mask] = correct
            done[answer_mask] = True
        obs = self._encode_obs(self._visible, self._revealed, self._revealed_content, self._target)
        valid = torch.ones(*shape, self.NUM_ACTIONS, dtype=torch.bool, device=actions.device)
        valid[..., 0] = ~self._visible.any(dim=-1)
        for i in range(self.num_files):
            valid[..., i + 1] = self._visible[..., i] & ~self._revealed[..., i]
            valid[..., i + 6] = self._visible[..., i]
        valid[done] = False
        return TurnBatch(obs=obs, env_target=env_target, reward=reward, done=done, valid_actions=valid)


class TwoPhaseMemoryEnv:
    """Two-phase memory task that truly requires recurrent state.

    Phase 1 (READ): Agent sees which files exist and their CONTENTS when read.
      Observation: [5 file_exists, 5 file_read, 5 content_visible, 5 target_onehot]
      Actions: READ_0..4 (5 actions)
      The OBSERVATION SHOWS content that was just read (via content_visible flags).

    Phase 2 (ANSWER): All content is HIDDEN. Agent must remember from Phase 1.
      Observation: [5 file_exists, 5 file_read, 5 zeros, 5 target_onehot]
      Actions: ANSWER_0..4 (5 actions)
      NO content information in observation — MUST use memory.

    Total actions: 10 (5 READ + 5 ANSWER).
    Env response classes: 6 (NULL + 5 content words).
    """

    NUM_ACTIONS = 10
    NUM_ENV_CLASSES = 6
    OBS_DIM = 20
    MAX_TURNS = 8
    READ_PHASE_LEN = 3

    def __init__(self, num_files: int = 5, num_words: int = 5, seed: int = 42):
        self.num_files = num_files
        self.num_words = num_words
        self._seed = seed
        self.rng = None

    def _random_state(self, shape: Tuple[int, ...], device: torch.device) -> Tuple[Tensor, Tensor]:
        if self.rng is None or self.rng.device != device:
            self.rng = torch.Generator(device=device)
            self.rng.manual_seed(self._seed)
        contents = torch.randint(0, self.num_words, (*shape, self.num_files), device=device, generator=self.rng)
        target = torch.randint(0, self.num_words, shape, device=device, generator=self.rng)
        target_in = (contents == target.unsqueeze(-1)).any(dim=-1)
        needs_fix = ~target_in
        if needs_fix.any():
            fix_idx = torch.randint(0, self.num_files, shape, device=device, generator=self.rng)
            flat_needs = needs_fix.reshape(-1)
            flat_contents = contents.reshape(-1, self.num_files)
            flat_fix = fix_idx.reshape(-1)
            flat_target = target.reshape(-1)
            for i in range(flat_needs.shape[0]):
                if flat_needs[i]:
                    flat_contents[i, flat_fix[i]] = flat_target[i]
            contents = flat_contents.reshape(*shape, self.num_files)
        return contents, target

    def _encode_obs(self, file_read: Tensor, content_visible: Tensor, target: Tensor, in_read_phase: Tensor) -> Tensor:
        obs_shape = file_read.shape[:-1]
        exists = torch.ones_like(file_read, dtype=torch.float32).reshape(-1, self.num_files)
        read_f = file_read.float().reshape(-1, self.num_files)
        content_f = content_visible.float().reshape(-1, self.num_files)
        target_oh = torch.zeros(exists.shape[0], self.num_words, device=file_read.device)
        target_flat = target.reshape(-1, 1)
        target_oh.scatter_(1, target_flat, 1.0)
        return torch.cat([exists, read_f, content_f, target_oh], dim=-1).reshape(*obs_shape, self.OBS_DIM)

    def reset(self, batch_size: int, device: torch.device, num_candidates: int = 1) -> TurnBatch:
        shape = (batch_size, num_candidates) if num_candidates > 1 else (batch_size,)
        contents, target = self._random_state(shape, device)
        file_read = torch.zeros(*shape, self.num_files, dtype=torch.bool, device=device)
        file_content = torch.full((*shape, self.num_files), -1, dtype=torch.long, device=device)
        content_visible = torch.zeros(*shape, self.num_files, dtype=torch.bool, device=device)
        in_read_phase = torch.ones(*shape, dtype=torch.bool, device=device)
        self._state_shape = shape
        self._contents = contents
        self._target = target
        self._file_read = file_read
        self._file_content = file_content
        self._content_visible = content_visible
        self._in_read_phase = in_read_phase
        self._turns_taken = torch.zeros(*shape, dtype=torch.long, device=device)
        obs = self._encode_obs(file_read, content_visible, target, in_read_phase)
        valid = torch.ones(*shape, self.NUM_ACTIONS, dtype=torch.bool, device=device)
        for i in range(self.num_files):
            valid[..., i] = ~file_read[..., i]
            valid[..., i + self.num_files] = False
        return TurnBatch(obs=obs, env_target=torch.zeros(*shape, dtype=torch.long, device=device), reward=torch.zeros(*shape, device=device), done=torch.zeros(*shape, dtype=torch.bool, device=device), valid_actions=valid)

    def step(self, actions: Tensor) -> TurnBatch:
        shape = self._state_shape
        read_mask = (actions >= 0) & (actions < self.num_files)
        answer_mask = (actions >= self.num_files) & (actions < self.num_files * 2)
        env_target = torch.zeros(*shape, dtype=torch.long, device=actions.device)
        reward = torch.zeros(*shape, device=actions.device)
        done = torch.zeros(*shape, dtype=torch.bool, device=actions.device)
        if read_mask.any():
            file_idx = actions[read_mask].long()
            idx_tuple = torch.where(read_mask)
            contents_idx = self._contents[idx_tuple]
            content = contents_idx[torch.arange(len(idx_tuple[0])), file_idx]
            reveal_idx = (*idx_tuple, file_idx)
            self._file_read[reveal_idx] = True
            self._file_content[reveal_idx] = content
            self._content_visible[reveal_idx] = True
            env_target[read_mask] = content + 1
        if answer_mask.any():
            file_idx = (actions[answer_mask] - self.num_files).long()
            idx_tuple = torch.where(answer_mask)
            contents_idx = self._contents[idx_tuple]
            content = contents_idx[torch.arange(len(idx_tuple[0])), file_idx]
            target_idx = self._target[idx_tuple]
            correct = (content == target_idx).float()
            reward[answer_mask] = correct
            done[answer_mask] = True
        self._turns_taken = self._turns_taken + 1
        in_read = self._turns_taken < self.READ_PHASE_LEN
        if not in_read.any():
            self._content_visible[:] = False
        obs = self._encode_obs(self._file_read, self._content_visible, self._target, in_read)
        valid = torch.ones(*shape, self.NUM_ACTIONS, dtype=torch.bool, device=actions.device)
        for i in range(self.num_files):
            valid[..., i] = in_read & ~self._file_read[..., i]
            valid[..., i + self.num_files] = ~in_read
        valid[done] = False
        done = done | (self._turns_taken >= self.MAX_TURNS)
        return TurnBatch(obs=obs, env_target=env_target, reward=reward, done=done, valid_actions=valid)
