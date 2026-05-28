import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LEGACY = _PROJECT_ROOT / "legacy"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from sef_gram.tokenizer import CharacterTokenizer
from experiment_terminal_memory import (
    EFLACharacterModel,
    SyntheticTerminalEmulator,
    HFTerminalStreamer
)

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
