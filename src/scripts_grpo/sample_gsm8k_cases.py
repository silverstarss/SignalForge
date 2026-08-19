#!/usr/bin/env python3
"""Save GSM8K model outputs before or after training.

The script expects veRL-style GSM8K parquet files from examples/data_preprocess/gsm8k.py.
It writes JSONL rows with prompt, ground truth, response, extracted answer, strict/flexible
correctness, response token length, and generation latency.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text: str, strict: bool) -> str | None:
    tail = text[-400:]
    if strict:
        matches = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)", tail)
    else:
        matches = re.findall(r"(-?[0-9][0-9,]*(?:\.[0-9]+)?)", tail)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def normalize_answer(value: Any) -> str:
    return str(value).replace(",", "").strip()


def prompt_text(tokenizer: Any, prompt: Any) -> str:
    messages = prompt
    if isinstance(prompt, str):
        try:
            messages = json.loads(prompt)
        except json.JSONDecodeError:
            messages = [{"role": "user", "content": prompt}]
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/gsm8k/test.parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype: Any = "auto"
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    df = pd.read_parquet(args.data).head(args.limit)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        prompt = prompt_text(tokenizer, row["prompt"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.time()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_s = time.time() - start
        response_ids = output[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
        response_tokens = int(response_ids.shape[1])
        gt = normalize_answer(row["reward_model"]["ground_truth"])
        strict_answer = extract_answer(response, strict=True)
        flexible_answer = extract_answer(response, strict=False)
        rows.append(
            {
                "index": int(idx),
                "question": row["extra_info"]["question"],
                "ground_truth": gt,
                "response": response,
                "response_tokens": response_tokens,
                "latency_s": latency_s,
                "strict_answer": strict_answer,
                "flexible_answer": flexible_answer,
                "strict_correct": strict_answer == gt,
                "flexible_correct": flexible_answer == gt,
                "format_error": strict_answer is None,
            }
        )
        print(f"sampled {len(rows)}/{len(df)}")

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    strict_acc = sum(r["strict_correct"] for r in rows) / len(rows)
    flex_acc = sum(r["flexible_correct"] for r in rows) / len(rows)
    fmt_err = sum(r["format_error"] for r in rows) / len(rows)
    print(json.dumps({"strict_accuracy": strict_acc, "flexible_accuracy": flex_acc, "format_error_rate": fmt_err}, indent=2))


if __name__ == "__main__":
    main()

