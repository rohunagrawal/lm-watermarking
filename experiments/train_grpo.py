"""Entry-point script to fine-tune a small language model with GRPO."""

from __future__ import annotations

import argparse
import math
from typing import Dict, Iterator, Optional

import torch

try:
    import wandb
except ImportError as exc:  # pragma: no cover - dependency availability
    raise ImportError(
        "train_grpo requires the 'wandb' package. Install it with 'pip install wandb'."
    ) from exc

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - dependency availability
    raise ImportError(
        "train_grpo requires the 'datasets' package. Install it with 'pip install datasets'."
    ) from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - dependency availability
    raise ImportError(
        "train_grpo requires the 'transformers' package. Install it with 'pip install transformers'."
    ) from exc

from experiments.grpo_trainer import GRPOConfig, GRPOTrainer, collate_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face model identifier to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        default="gsm8k",
        help="Dataset name on the Hugging Face Hub used for RL fine-tuning.",
    )
    parser.add_argument(
        "--dataset-config",
        default="main",
        help="Optional dataset configuration (e.g. subset) to load.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load for training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=64,
        help="Maximum number of samples pulled from the dataset (<=0 keeps all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Number of rollouts per optimisation step."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens generated for each answer.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate used by the AdamW optimiser.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum number of training steps to run.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for model execution (e.g. 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling sampling operations.",
    )
    parser.add_argument(
        "--wandb-project",
        default="grpo-training",
        help="Weights & Biases project name for logging.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name. If None, auto-generated.",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="Weights & Biases entity (username or team name).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    return parser.parse_args()


def format_prompt(question: str) -> str:
    return f"Solve the following grade-school math problem and answer succinctly.\n\nQuestion: {question}\nAnswer:"  # noqa: E501


def extract_numeric_answer(answer: str) -> Optional[str]:
    if "####" in answer:
        answer = answer.split("####", 1)[-1]
    answer = answer.strip()
    if not answer:
        return None
    # keep just the last line, which usually contains the numeric answer in GSM8K
    return answer.splitlines()[-1].strip()


def compute_reward(response: str, reference_answer: str) -> torch.Tensor:
    response = response.strip().splitlines()[-1]
    target = extract_numeric_answer(reference_answer)
    reward = 1.0 if target and target in response else 0.0
    return torch.tensor(reward, dtype=torch.float32)


@torch.no_grad()
def generate_rollout(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    reference_answer: str,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, torch.Tensor]:
    tokenized = tokenizer(prompt, return_tensors="pt")
    tokenized = {k: v.to(device) for k, v in tokenized.items()}
    prompt_length = tokenized["input_ids"].size(1)

    generation = model.generate(
        **tokenized,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    full_input_ids = generation[0].cpu()
    attention_mask = torch.ones_like(full_input_ids)
    response_mask = torch.zeros_like(full_input_ids)
    response_mask[prompt_length:] = 1

    response_text = tokenizer.decode(
        full_input_ids[prompt_length:], skip_special_tokens=True
    )
    reward = compute_reward(response_text, reference_answer)

    return {
        "input_ids": full_input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
        "rewards": reward,
    }


def rollout_dataloader(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset,
    device: torch.device,
    batch_size: int,
    max_new_tokens: int,
) -> Iterator[Dict[str, torch.Tensor]]:
    for start in range(0, len(dataset), batch_size):
        model.eval()
        samples = []
        end = min(start + batch_size, len(dataset))
        for index in range(start, end):
            example = dataset[index]
            prompt = format_prompt(example["question"])
            rollout = generate_rollout(
                model, tokenizer, prompt, example["answer"], device, max_new_tokens
            )
            samples.append(rollout)
        yield collate_fn(samples)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config={
                "model_name": args.model_name,
                "dataset_name": args.dataset_name,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "max_samples": args.max_samples,
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "learning_rate": args.learning_rate,
                "max_steps": args.max_steps,
                "device": args.device,
                "seed": args.seed,
            }
        )

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.special_tokens_map.get("bos_token", 0)

    policy_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    reference_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    reference_model.eval()
    policy_model.config.pad_token_id = tokenizer.pad_token_id
    reference_model.config.pad_token_id = tokenizer.pad_token_id

    dataset_kwargs = {"split": args.split}
    if args.dataset_config:
        dataset_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset_name, **dataset_kwargs)
    if args.max_samples > 0:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty; adjust dataset arguments.")

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    trainer = GRPOTrainer(
        policy_model,
        tokenizer,
        optimizer,
        reference_model=reference_model,
        config=GRPOConfig(),
        device=device,
        use_wandb=not args.no_wandb,
    )

    steps = math.ceil(len(dataset) / args.batch_size)
    steps = min(steps, args.max_steps)

    for step, batch in zip(
        range(steps),
        rollout_dataloader(
            policy_model,
            tokenizer,
            dataset,
            device,
            args.batch_size,
            args.max_new_tokens,
        ),
    ):
        policy_model.train()
        stats = trainer.step(batch)
        print(f"step={step} | reward_mean={stats['reward_mean']:.3f} | kl={stats['kl_mean']:.3f}")

    print("Training finished")
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
