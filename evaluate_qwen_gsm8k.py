#!/usr/bin/env python
"""Evaluate pass@k for a Qwen language model on GSM8K."""
import argparse
import json
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SampleResult:
    """Stores generation results for a single GSM8K problem."""

    question: str
    reference_answer: str
    generations: List[Dict[str, Optional[str]]] = field(default_factory=list)
    pass_at_k: Dict[int, float] = field(default_factory=dict)


ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
MODEL_ANSWER_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def parse_gsm8k_answer(answer: str) -> Optional[str]:
    """Extract the canonical numeric answer string from a GSM8K reference solution."""

    match = ANSWER_PATTERN.search(answer)
    return match.group(1) if match else None


def parse_model_answer(completion: str) -> Optional[str]:
    """Attempt to extract the final numeric answer from a model completion."""

    # Search the last 3 lines for a numeric answer to reduce spurious matches.
    lines = [line.strip() for line in completion.strip().splitlines() if line.strip()]
    for line in reversed(lines[-3:]):
        match = MODEL_ANSWER_PATTERN.search(line)
        if match:
            return match.group(1)
    return None


def is_answer_correct(predicted: Optional[str], reference: Optional[str], atol: float = 1e-6) -> bool:
    """Compare predicted and reference answers allowing for floating point tolerance."""

    if predicted is None or reference is None:
        return False

    try:
        predicted_value = float(predicted)
        reference_value = float(reference)
    except ValueError:
        return False

    return math.isclose(predicted_value, reference_value, rel_tol=0.0, abs_tol=atol)


def compute_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Compute the pass@k estimate using the unbiased estimator from the Codex paper."""

    if k <= 0:
        raise ValueError("k must be positive")

    if num_correct == 0:
        return 0.0

    if num_samples < k:
        return 1.0 if num_correct > 0 else 0.0

    failures = num_samples - num_correct
    numerator = math.comb(failures, k) if failures >= k else 0
    denominator = math.comb(num_samples, k)
    return 1.0 - (numerator / denominator)


def build_prompt(question: str, system_prompt: str) -> str:
    user_content = (
        "Solve the following grade school math word problem step-by-step. "
        "Respond with your reasoning and finish your reply with a line starting with 'Answer:' "
        "followed by the final numeric result.\n\n"
        f"Problem: {question.strip()}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_completions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    seed: int,
) -> List[str]:
    """Generate ``num_samples`` completions for a single question."""

    completions: List[str] = []

    for sample_idx in range(num_samples):
        generator = torch.Generator(device=device).manual_seed(seed + sample_idx)
        messages = build_prompt(question, system_prompt)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        output = model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            generator=generator,
        )

        generated_ids = output[0, encoded["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        completions.append(completion)

    return completions


def evaluate_model(args: argparse.Namespace) -> Dict[str, float]:
    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.load_in_half and torch.cuda.is_available() else None,
        device_map="auto" if torch.cuda.is_available() and not args.cpu else None,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    aggregate_pass_at_k = {k: [] for k in args.pass_k}
    sample_results: List[SampleResult] = []

    for idx, example in enumerate(dataset):
        reference_answer = parse_gsm8k_answer(example["answer"])
        completions = generate_completions(
            model=model,
            tokenizer=tokenizer,
            question=example["question"],
            system_prompt=args.system_prompt,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            seed=args.seed + idx * args.num_samples,
        )

        per_sample = SampleResult(
            question=example["question"],
            reference_answer=reference_answer or "",
        )

        num_correct = 0
        for completion in completions:
            predicted_answer = parse_model_answer(completion)
            correct = is_answer_correct(predicted_answer, reference_answer)
            if correct:
                num_correct += 1
            per_sample.generations.append(
                {
                    "completion": completion,
                    "predicted_answer": predicted_answer,
                    "is_correct": correct,
                }
            )

        for k in args.pass_k:
            score = compute_pass_at_k(args.num_samples, num_correct, k)
            per_sample.pass_at_k[k] = score
            aggregate_pass_at_k[k].append(score)

        sample_results.append(per_sample)

        if args.verbose:
            print(f"Problem {idx + 1}: {num_correct}/{args.num_samples} correct")
            for k in sorted(args.pass_k):
                print(f"  pass@{k}: {per_sample.pass_at_k[k]:.3f}")

    averaged_scores = {
        f"pass@{k}": float(sum(scores) / len(scores)) if scores else 0.0
        for k, scores in aggregate_pass_at_k.items()
    }

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "config": vars(args),
                    "averaged_scores": averaged_scores,
                    "samples": [
                        {
                            "question": result.question,
                            "reference_answer": result.reference_answer,
                            "generations": result.generations,
                            "pass_at_k": result.pass_at_k,
                        }
                        for result in sample_results
                    ],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

    return averaged_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen1.5-0.5B-Chat",
        help="Name of the Hugging Face model to evaluate.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"]
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit the number of GSM8K problems to evaluate (None evaluates full split).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of stochastic completions to generate per problem.",
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Values of k for pass@k computation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a careful math tutor who explains their reasoning clearly and concisely.",
        help="System prompt used for chat-based Qwen models.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional path to save detailed generation records as JSON.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--load-in-half",
        action="store_true",
        help="Load the model weights in float16 precision when CUDA is available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-problem metrics during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    scores = evaluate_model(args)

    print("Averaged pass@k scores:")
    for k, score in sorted(scores.items()):
        print(f"  {k}: {score:.4f}")


if __name__ == "__main__":
    main()
