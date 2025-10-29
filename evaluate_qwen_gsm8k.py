#!/usr/bin/env python
"""Evaluate pass@k for a Qwen language model on GSM8K."""
import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from extended_watermark_processor import WatermarkLogitsProcessor
import wandb


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


def _repeat_batch(encoded: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    """Repeat tokenized inputs along the batch dimension."""

    if batch_size == 1:
        return encoded

    repeated: Dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        repeats = (batch_size,) + (1,) * (value.dim() - 1)
        repeated[key] = value.repeat(repeats)
    return repeated


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
    batch_size: int,
    answer: str
) -> List[str]:
    """Generate ``num_samples`` completions for a single question."""

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

    completions: List[str] = []
    total_batches = math.ceil(num_samples / batch_size)

    for batch_idx in range(total_batches):
        current_batch_size = min(batch_size, num_samples - len(completions))
        print(
            "Generating batch"
            f" {batch_idx + 1} of {total_batches} (normal generation, batch size {current_batch_size})"
        )
        start_time = time.time()

        batch_inputs = _repeat_batch(encoded, current_batch_size)

        output = model.generate(
            **batch_inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = output[:, encoded["input_ids"].shape[1]:]
        batch_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions.extend(completion.strip() for completion in batch_completions)

        end_time = time.time()
        generation_time = end_time - start_time
        if batch_completions:
            print("Question:")
            print(question)
            print("Ground-Truth Answer:")
            print(answer)
            print("Example completion from this batch:")
            print(batch_completions[0])
        print(
            "Normal generation batch"
            f" {batch_idx + 1} completed in {generation_time:.2f} seconds"
        )

    return completions


def generate_completions_with_watermark(
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
    watermark_gamma: float,
    watermark_delta: float,
    watermark_seeding_scheme: str,
    watermark_seed: int,
    batch_size: int,
    answer: str
) -> List[str]:
    """Generate ``num_samples`` completions that apply the repository watermark."""

    vocab_ids = list(tokenizer.get_vocab().values())

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

    completions: List[str] = []
    total_batches = math.ceil(num_samples / batch_size)

    for batch_idx in range(total_batches):
        current_batch_size = min(batch_size, num_samples - len(completions))
        print(
            "Generating batch"
            f" {batch_idx + 1} of {total_batches} (watermark generation, batch size {current_batch_size})"
        )
        start_time = time.time()

        watermark_processor = WatermarkLogitsProcessor(
            vocab=vocab_ids,
            gamma=watermark_gamma,
            delta=watermark_delta,
            seeding_scheme=watermark_seeding_scheme,
        )
        watermark_processor.rng = torch.Generator(device=device)
        watermark_processor.rng.manual_seed(watermark_seed + len(completions))

        logits_processor = LogitsProcessorList([watermark_processor])
        batch_inputs = _repeat_batch(encoded, current_batch_size)
    
        output = model.generate(
            **batch_inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
        )

        generated_ids = output[:, encoded["input_ids"].shape[1]:]
        batch_completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        completions.extend(completion.strip() for completion in batch_completions)

        end_time = time.time()
        generation_time = end_time - start_time
        # INSERT_YOUR_CODE
        if batch_completions:
            print("Question:")
            print(question)
            print("Ground-Truth Answer:")
            print(answer)
            print("Example completion from this batch:")
            print(batch_completions[0])
        print(
            "Watermark generation batch"
            f" {batch_idx + 1} completed in {generation_time:.2f} seconds"
        )

    return completions


def evaluate_model(args: argparse.Namespace) -> Dict[str, float]:
    # WANDB SETUP
    if not args.disable_watermark:
        run_name = f"qwen-gsm8k-{args.model_name}-gamma{args.watermark_gamma}-delta{args.watermark_delta}"
    else:
        run_name = f"qwen-gsm8k-{args.model_name}"
    wandb.init(project="qwen-gsm8k-eval", config=vars(args), name=run_name)
    run_table = wandb.Table(columns=["idx", "question", "reference_answer", "completion", "predicted_answer", "is_correct", "watermark_applied"], log_mode="MUTABLE")

    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    if args.generation_batch_size is None:
        args.generation_batch_size = args.num_samples
    if args.generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be a positive integer")

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
        if args.disable_watermark:
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
                batch_size=args.generation_batch_size,
                answer=reference_answer,
            )
        else:
            completions = generate_completions_with_watermark(
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
                watermark_gamma=args.watermark_gamma,
                watermark_delta=args.watermark_delta,
                watermark_seeding_scheme=args.watermark_seeding_scheme,
                watermark_seed=args.watermark_seed + idx * args.num_samples,
                batch_size=args.generation_batch_size,
                answer=reference_answer
            )

        per_sample = SampleResult(
            question=example["question"],
            reference_answer=reference_answer or "",
        )

        num_correct = 0
        unique_predicted_answers = set()
        for completion in completions:
            predicted_answer = parse_model_answer(completion)
            if predicted_answer is not None:
                unique_predicted_answers.add(predicted_answer)
            correct = is_answer_correct(predicted_answer, reference_answer)
            if correct:
                num_correct += 1
            per_sample.generations.append(
                {
                    "completion": completion,
                    "predicted_answer": predicted_answer,
                    "is_correct": correct,
                    "watermark_applied": not args.disable_watermark,
                }
            )
            # WANDB: log generation to table
            run_table.add_data(
                idx,
                example["question"],
                reference_answer,
                completion,
                predicted_answer,
                correct,
                not args.disable_watermark
            )
            wandb.log({"generations": run_table})
        num_unique_predicted_answers = len(unique_predicted_answers)
        wandb.log({"num_unique_predicted_answers": num_unique_predicted_answers, "problem_idx": idx})

        for k in args.pass_k:
            score = compute_pass_at_k(args.num_samples, num_correct, k)
            per_sample.pass_at_k[k] = score
            aggregate_pass_at_k[k].append(score)
            # WANDB: log per-problem pass@k metric
            wandb.log({f"pass@{k}_problem": score, "problem_idx": idx})

        sample_results.append(per_sample)

        if args.verbose:
            print(f"Problem {idx + 1}: {num_correct}/{args.num_samples} correct")
            for k in sorted(args.pass_k):
                print(f"  pass@{k}: {per_sample.pass_at_k[k]:.3f}")

        averaged_scores = {
            f"pass@{k}": float(sum(scores) / len(scores)) if scores else 0.0
            for k, scores in aggregate_pass_at_k.items()
        }

        print(averaged_scores)

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

    # WANDB: log averages and generations table
    averaged_scores = {
        f"pass@{k}": float(sum(scores) / len(scores)) if scores else 0.0
        for k, scores in aggregate_pass_at_k.items()
    }
    wandb.log({**averaged_scores, "final_step": True})
    wandb.log({"generations": run_table})
    wandb.finish()

    return averaged_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
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
        default=[1, 2, 4],
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
        "--generation-batch-size",
        type=int,
        default=None,
        help="Number of completions to generate simultaneously.",
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
    parser.add_argument(
        "--disable-watermark",
        action="store_true",
        help="Skip applying the watermark during generation.",
    )
    parser.add_argument(
        "--watermark-gamma",
        type=float,
        default=0.25,
        help="Fraction of the vocabulary used for the watermark greenlist.",
    )
    parser.add_argument(
        "--watermark-delta",
        type=float,
        default=0.5,
        help="Logit bias added to watermark greenlist tokens.",
    )
    parser.add_argument(
        "--watermark-seeding-scheme",
        default="lefthash",
        help="Seeding scheme used by the watermark processor.",
    )
    parser.add_argument(
        "--watermark-seed",
        type=int,
        default=0,
        help="Base seed for the watermark RNG.",
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
