#!/usr/bin/env python
"""Evaluate pass@k for a Qwen language model on GSM8K with retry attempts."""
import argparse
import json
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


def build_prompt(question: str, system_prompt: str) -> List[Dict[str, str]]:
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


def _generate_attempt(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generator_seed: int,
    watermark_config: Optional[Tuple[float, float, str, int]],
    vocab_ids: Optional[List[int]],
) -> str:
    """Generate a single assistant reply for the provided conversation."""

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

    generator = torch.Generator(device=device)
    generator.manual_seed(generator_seed)

    logits_processor = None
    if watermark_config is not None:
        gamma, delta, seeding_scheme, watermark_seed = watermark_config
        watermark_processor = WatermarkLogitsProcessor(
            vocab=vocab_ids if vocab_ids is not None else list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
        )
        watermark_processor.rng = torch.Generator(device=device)
        watermark_processor.rng.manual_seed(watermark_seed)
        logits_processor = LogitsProcessorList([watermark_processor])

    output = model.generate(
        **encoded,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
        logits_processor=logits_processor,
    )

    generated_ids = output[:, encoded["input_ids"].shape[1]:]
    completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return completion.strip()


def generate_with_retries(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str,
    reference_answer: Optional[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    max_attempts: int,
    retry_prompt: str,
    watermark_params: Optional[Tuple[float, float, str, int]],
    vocab_ids: Optional[List[int]],
) -> Tuple[List[Dict[str, Optional[str]]], bool]:
    """Run a conversation with retries until success or attempts are exhausted."""

    messages = build_prompt(question, system_prompt)
    attempts: List[Dict[str, Optional[str]]] = []
    success = False

    for attempt_idx in range(max_attempts):
        start_time = time.time()
        watermark_config = None
        if watermark_params is not None:
            gamma, delta, seeding_scheme, base_watermark_seed = watermark_params
            watermark_config = (
                gamma,
                delta,
                seeding_scheme,
                base_watermark_seed + attempt_idx,
            )

        completion = _generate_attempt(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generator_seed=base_seed + attempt_idx,
            watermark_config=watermark_config,
            vocab_ids=vocab_ids,
        )
        elapsed = time.time() - start_time

        predicted_answer = parse_model_answer(completion)
        correct = is_answer_correct(predicted_answer, reference_answer)

        attempts.append(
            {
                "attempt": attempt_idx + 1,
                "completion": completion,
                "predicted_answer": predicted_answer,
                "is_correct": correct,
                "watermark_applied": watermark_params is not None,
                "generation_time": elapsed,
            }
        )

        messages.append({"role": "assistant", "content": completion})

        if correct:
            success = True
            break

        if attempt_idx < max_attempts - 1:
            messages.append({"role": "user", "content": retry_prompt})

    return attempts, success


def evaluate_model(args: argparse.Namespace) -> Dict[str, float]:
    # WANDB SETUP
    if not args.disable_watermark:
        run_name = (
            "qwen-gsm8k-retries-"
            f"{args.model_name}-gamma{args.watermark_gamma}-delta{args.watermark_delta}"
        )
    else:
        run_name = f"qwen-gsm8k-retries-{args.model_name}"
    wandb.init(project="qwen-gsm8k-eval", config=vars(args), name=run_name)
    run_table = wandb.Table(
        columns=[
            "idx",
            "sample_index",
            "attempt",
            "question",
            "reference_answer",
            "completion",
            "predicted_answer",
            "is_correct",
            "watermark_applied",
        ],
        log_mode="MUTABLE",
    )

    dataset = load_dataset("gsm8k", "main", split=args.split)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    if args.max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

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

    vocab_ids = list(tokenizer.get_vocab().values())

    aggregate_pass_at_k = {k: [] for k in args.pass_k}
    sample_results: List[SampleResult] = []

    for idx, example in enumerate(dataset):
        reference_answer = parse_gsm8k_answer(example["answer"])
        per_sample = SampleResult(
            question=example["question"],
            reference_answer=reference_answer or "",
        )

        num_correct = 0
        unique_predicted_answers = set()

        for sample_idx in range(args.num_samples):
            base_seed = args.seed + idx * args.num_samples + sample_idx * args.max_attempts
            watermark_seed = (
                args.watermark_seed + idx * args.num_samples + sample_idx * args.max_attempts
            )
            watermark_params = None
            if not args.disable_watermark:
                watermark_params = (
                    args.watermark_gamma,
                    args.watermark_delta,
                    args.watermark_seeding_scheme,
                    watermark_seed,
                )

            attempts, success = generate_with_retries(
                model=model,
                tokenizer=tokenizer,
                question=example["question"],
                system_prompt=args.system_prompt,
                reference_answer=reference_answer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                base_seed=base_seed,
                max_attempts=args.max_attempts,
                retry_prompt=args.retry_prompt,
                watermark_params=watermark_params,
                vocab_ids=vocab_ids,
            )

            for attempt in attempts:
                if attempt["predicted_answer"] is not None:
                    unique_predicted_answers.add(attempt["predicted_answer"])
                record = {
                    "sample_index": sample_idx,
                    **attempt,
                }
                per_sample.generations.append(record)
                run_table.add_data(
                    idx,
                    sample_idx,
                    attempt["attempt"],
                    example["question"],
                    reference_answer,
                    attempt["completion"],
                    attempt["predicted_answer"],
                    attempt["is_correct"],
                    attempt["watermark_applied"],
                )
                wandb.log({"generations": run_table})

            if success:
                num_correct += 1

        num_unique_predicted_answers = len(unique_predicted_answers)
        wandb.log({"num_unique_predicted_answers": num_unique_predicted_answers, "problem_idx": idx})

        for k in args.pass_k:
            score = compute_pass_at_k(args.num_samples, num_correct, k)
            per_sample.pass_at_k[k] = score
            aggregate_pass_at_k[k].append(score)
            wandb.log({f"pass@{k}_problem": score, "problem_idx": idx})

        sample_results.append(per_sample)

        if args.verbose:
            print(f"Problem {idx + 1}: {num_correct}/{args.num_samples} conversations solved")
            for k in sorted(args.pass_k):
                print(f"  pass@{k}: {per_sample.pass_at_k[k]:.3f}")

        averaged_scores = {
            f"pass@{k}": float(sum(scores) / len(scores)) if scores else 0.0
            for k, scores in aggregate_pass_at_k.items()
        }

        print(averaged_scores)

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
        choices=["train", "test"],
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
        help="Number of independent conversations to run per problem.",
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
        "--system-prompt",
        default="You are a careful math tutor who explains their reasoning clearly and concisely.",
        help="System prompt used for chat-based Qwen models.",
    )
    parser.add_argument(
        "--retry-prompt",
        default="Your answer was wrong. Please try something different.",
        help="Prompt appended after incorrect attempts.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts per conversation.",
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
