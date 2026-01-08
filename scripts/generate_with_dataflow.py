#!/usr/bin/env python3
"""
Generate Long-Context Hallucination Detection Dataset using DataFlow operators.

Usage:
    python scripts/generate_with_dataflow.py --num-samples 50
"""

import sys
sys.path.insert(0, ".")

import json
import random
import re
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from openai import OpenAI

# Import DataFlow operators
from dataflow.operators.hallucination_detection import (
    LongContextFilterOperator,
    HallucinationInjectionOperator,
)


# Hallucination injection prompts
EVIDENT_CONFLICT = """Modify the answer to contain a DIRECT CONTRADICTION of a fact in the reference. Mark with <hal>...</hal>.
Reference excerpt: {reference}
Original answer: {answer}
Return ONLY the modified answer with <hal> tags."""

EVIDENT_BASELESS = """Add a FABRICATED fact not in the reference. Mark with <hal>...</hal>.
Reference excerpt: {reference}
Original answer: {answer}
Return ONLY the modified answer with <hal> tags."""


def parse_hal_tags(text):
    """Parse <hal>...</hal> tags to extract span positions."""
    labels = []
    for match in re.finditer(r"<hal>(.*?)</hal>", text, re.DOTALL):
        labels.append({"text": match.group(1), "label": "hallucinated"})
    
    clean = re.sub(r"<hal>(.*?)</hal>", r"\1", text, flags=re.DOTALL)
    
    for label in labels:
        start = clean.find(label["text"])
        if start >= 0:
            label["start"] = start
            label["end"] = start + len(label["text"])
    
    return labels, clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=24000)
    parser.add_argument("--hal-ratio", type=float, default=0.5)
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--output-dir", default="output/longcontext_haldetect_dataflow")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    print("Loading NarrativeQA dataset...")
    ds = load_dataset("deepmind/narrativeqa", split="test")
    
    # Convert to DataFrame
    print("Converting to DataFrame...")
    samples = []
    for i, item in enumerate(tqdm(ds, desc="Processing")):
        if len(samples) >= args.num_samples * 3:  # Get more for filtering
            break
        
        doc = item.get("document", {})
        text = doc.get("text", "") if isinstance(doc, dict) else ""
        if not text:
            continue
        
        question = item.get("question", {})
        q_text = question.get("text", "") if isinstance(question, dict) else str(question)
        
        samples.append({
            "id": f"narrativeqa_{i}",
            "document": text[:50000],
            "question": q_text,
            "source": "narrativeqa",
        })
    
    df = pd.DataFrame(samples)
    print(f"Loaded {len(df)} samples")
    
    # Step 1: Filter by token count
    print("\n" + "=" * 50)
    print(f"STEP 1: Filtering by token count ({args.min_tokens}-{args.max_tokens})")
    print("=" * 50)
    
    filtered_samples = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        tokens = len(tokenizer.encode(row["document"], add_special_tokens=True))
        if args.min_tokens <= tokens <= args.max_tokens:
            row_dict = row.to_dict()
            row_dict["num_tokens"] = tokens
            filtered_samples.append(row_dict)
            if len(filtered_samples) >= args.num_samples:
                break
    
    if len(filtered_samples) == 0:
        print("No samples in range, trying 4K-16K...")
        for idx, row in df.iterrows():
            tokens = len(tokenizer.encode(row["document"], add_special_tokens=True))
            if 4000 <= tokens <= 16000:
                row_dict = row.to_dict()
                row_dict["num_tokens"] = tokens
                filtered_samples.append(row_dict)
                if len(filtered_samples) >= args.num_samples:
                    break
    
    filtered_df = pd.DataFrame(filtered_samples)
    print(f"Filtered: {len(filtered_df)} samples")
    
    if len(filtered_df) > 0:
        min_tok = filtered_df["num_tokens"].min()
        max_tok = filtered_df["num_tokens"].max()
        print(f"Token range: {min_tok} - {max_tok}")
    
    # Step 2: Generate answers
    print("\n" + "=" * 50)
    print("STEP 2: Generating answers via vLLM")
    print("=" * 50)
    
    client = OpenAI(base_url=args.api_url, api_key="dummy")
    
    answers = []
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Generating"):
        doc = row["document"][:25000]
        question = row["question"]
        
        prompt = f"""Based on the following document, answer the question.

Document:
{doc}

Question: {question}

Answer:"""
        
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.3,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Error: {e}")
            answer = "Unable to generate answer."
        
        answers.append(answer)
    
    filtered_df["answer"] = answers
    print(f"Generated {len(answers)} answers")
    
    # Step 3: Inject hallucinations
    print("\n" + "=" * 50)
    print(f"STEP 3: Injecting hallucinations ({args.hal_ratio*100:.0f}%)")
    print("=" * 50)
    
    hal_types = ["Evident Conflict", "Evident Baseless Info"]
    hal_prompts = {
        "Evident Conflict": EVIDENT_CONFLICT,
        "Evident Baseless Info": EVIDENT_BASELESS,
    }
    
    n_to_inject = int(len(filtered_df) * args.hal_ratio)
    inject_indices = set(random.sample(range(len(filtered_df)), n_to_inject))
    
    results = []
    stats = {"total": 0, "injected": 0, "failed": 0, "by_type": {}}
    
    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Injecting"):
        result = row.to_dict()
        result["has_hallucination"] = False
        result["hallucination_type"] = None
        result["labels"] = []
        
        if idx in inject_indices:
            hal_type = random.choice(hal_types)
            reference = row["document"][:3000]
            prompt = hal_prompts[hal_type].format(reference=reference, answer=row["answer"])
            
            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                    temperature=0.7,
                )
                modified = response.choices[0].message.content.strip()
                
                if "<hal>" in modified:
                    labels, clean = parse_hal_tags(modified)
                    result["answer"] = clean
                    result["has_hallucination"] = True
                    result["hallucination_type"] = hal_type
                    result["labels"] = labels
                    stats["injected"] += 1
                    stats["by_type"][hal_type] = stats["by_type"].get(hal_type, 0) + 1
                else:
                    stats["failed"] += 1
            except Exception as e:
                print(f"  Injection error: {e}")
                stats["failed"] += 1
        
        stats["total"] += 1
        results.append(result)
    
    output_df = pd.DataFrame(results)
    print(f"\nStats: {stats}")
    
    # Step 4: Save dataset
    print("\n" + "=" * 50)
    print("STEP 4: Saving dataset")
    print("=" * 50)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = output_df.to_dict(orient="records")
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    n_hal = sum(1 for d in dataset if d["has_hallucination"])
    n_sup = sum(1 for d in dataset if not d["has_hallucination"])
    
    print(f"Saved {len(dataset)} samples to {output_dir}/dataset.json")
    print(f"  - Hallucinated: {n_hal}")
    print(f"  - Supported: {n_sup}")
    
    if len(output_df) > 0:
        min_tok = output_df["num_tokens"].min()
        max_tok = output_df["num_tokens"].max()
        print(f"  - Token range: {min_tok} - {max_tok}")
    
    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()

