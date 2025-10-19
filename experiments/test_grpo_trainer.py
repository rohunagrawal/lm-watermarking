"""Sanity tests for the custom GRPO trainer implementation."""

from __future__ import annotations

import copy
import pathlib
import sys
from types import SimpleNamespace

import torch
from torch import nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grpo_trainer import GRPOConfig, GRPOTrainer, collate_fn


class TinyLM(nn.Module):
    """A tiny causal language model used for smoke testing the trainer."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, use_cache=False):
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        return SimpleNamespace(logits=logits, last_hidden_state=hidden)


def build_batch(batch_size: int = 4, seq_len: int = 8, prompt_len: int = 3):
    vocab_size = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    response_mask = torch.zeros_like(input_ids)
    response_mask[:, prompt_len:] = 1
    rewards = torch.randn(batch_size)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "rewards": rewards,
    }


def run_single_step() -> None:
    torch.manual_seed(0)
    policy = TinyLM()
    reference = copy.deepcopy(policy)
    value_head = None
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    trainer = GRPOTrainer(
        policy_model=policy,
        tokenizer=None,
        optimizer=optimizer,
        reference_model=reference,
        value_head=value_head,
        config=GRPOConfig(beta=0.5, entropy_coef=0.01),
    )
    batch = build_batch()
    stats = trainer.step(batch)
    print("Single step stats:", stats)


def test_collate_fn() -> None:
    torch.manual_seed(0)
    samples = []
    for length in (4, 6, 5):
        sample = build_batch(batch_size=1, seq_len=length)["input_ids"].squeeze(0)
        attention = torch.ones(length, dtype=torch.bool)
        response_mask = torch.ones(length, dtype=torch.bool)
        reward = torch.tensor(1.0)
        samples.append(
            {
                "input_ids": sample,
                "attention_mask": attention,
                "response_mask": response_mask,
                "rewards": reward,
            }
        )
    batch = collate_fn(samples)
    for key, tensor in batch.items():
        print(f"{key}: shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    run_single_step()
    test_collate_fn()
