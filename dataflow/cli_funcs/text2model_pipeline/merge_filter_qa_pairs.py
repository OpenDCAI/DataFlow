#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path


def find_input_file(cache_base="./"):
    """Find input file relative to cache_base"""
    cache_path = Path(cache_base)
    if not cache_path.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path = caller_cwd / cache_path

    print(f"Cache base directory: {cache_path}")

    # 优先查找 text2qa_step_step3.json
    possible_paths = [
        cache_path / ".cache" / "gpu" / "text2qa_step_step3.json",
        cache_path / ".cache" / "gpu" / "batch_cleaning_step_step4.json",
        cache_path / "cache" / "gpu" / "batch_cleaning_step_step4.json",
        cache_path / "batch_cleaning_step_step4.json",
    ]

    print("Searching for input file...")
    for path in possible_paths:
        abs_path = path.resolve()
        if abs_path.exists():
            size = abs_path.stat().st_size
            print(f"Found input file: {abs_path} ({size} bytes)")
            return abs_path
        else:
            print(f"Not found: {abs_path}")

    print("Input file not found!")
    return None


def load_qa_data_from_files(data_items, input_file):
    """Load QA data from enhanced_chunk_path files"""
    all_qa_pairs = []

    for i, item in enumerate(data_items):
        print(f"Processing item {i + 1}/{len(data_items)}: ", end="")

        enhanced_path = item.get('enhanced_chunk_path')
        if not enhanced_path:
            print("Skip (no enhanced_chunk_path)")
            continue

        # Convert to absolute path
        if not os.path.isabs(enhanced_path):
            input_file_path = Path(input_file)
            cache_gpu_dir = input_file_path.parent
            cache_dir = cache_gpu_dir.parent
            project_root = cache_dir.parent
            clean_path = enhanced_path.lstrip('./')
            enhanced_path = project_root / clean_path
        else:
            enhanced_path = Path(enhanced_path)

        if not enhanced_path.exists():
            print(f"Skip (file not exists: {enhanced_path})")
            continue

        try:
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)

            chunk_qa_pairs = []
            if isinstance(enhanced_data, list):
                # enhanced_data is chunk list
                for chunk in enhanced_data:
                    if isinstance(chunk, dict) and 'qa_pairs' in chunk:
                        qa_data = chunk['qa_pairs']
                        if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
                            # Double nested: chunk['qa_pairs']['qa_pairs']
                            chunk_qa_pairs.extend(qa_data['qa_pairs'])
                        elif isinstance(qa_data, list):
                            # Single nested: chunk['qa_pairs'] is directly a list
                            chunk_qa_pairs.extend(qa_data)
            elif isinstance(enhanced_data, dict) and 'qa_pairs' in enhanced_data:
                # enhanced_data is single object
                qa_data = enhanced_data['qa_pairs']
                if isinstance(qa_data, dict) and 'qa_pairs' in qa_data:
                    chunk_qa_pairs = qa_data['qa_pairs']
                elif isinstance(qa_data, list):
                    chunk_qa_pairs = qa_data

            if chunk_qa_pairs and isinstance(chunk_qa_pairs, list):
                print(f"Found {len(chunk_qa_pairs)} QA pairs")
                all_qa_pairs.extend(chunk_qa_pairs)
            else:
                print("Skip (no valid QA data)")
                continue

        except Exception as e:
            print(f"Skip (read failed: {e})")
            continue

    return all_qa_pairs


def convert_to_alpaca(input_file, output_dir):
    """Convert QA data to Alpaca format for LlamaFactory"""
    print(f"Reading data file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Read main data file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(
            f"Successfully read main data, type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

    if not isinstance(data, list):
        print("Expected data in list format")
        return None

    # Instruction for the training data
    instruction = (
        "Please answer the following question based on the provided text content. "
        "Your response should:\n"
        "1. Provide accurate information from the source material\n"
        "2. Include relevant analysis and reasoning\n"
        "3. Reference specific details or examples when applicable\n"
        "4. Maintain clarity and precision in your explanation\n\n"
        "Focus on delivering factual, well-reasoned answers based on the text content."
    )

    print("Loading QA pairs from enhanced files...")
    all_qa_pairs = load_qa_data_from_files(data, input_file)

    if not all_qa_pairs:
        print("No QA pairs found! Please check data structure")
        return None

    print(f"Total found {len(all_qa_pairs)} QA pairs")

    # Process QA pairs
    results = []
    processed_pairs = 0

    for qa in all_qa_pairs:
        if not isinstance(qa, dict):
            continue

        # Extract question and answer
        question = ""
        answer_text = ""

        # Try different possible field names for question
        for q_field in ['question', 'Question', 'query', 'Query']:
            if q_field in qa and qa[q_field]:
                question = qa[q_field].strip()
                break

        # Try different possible field names for answer
        for a_field in ['answer', 'Answer', 'response', 'Response']:
            if a_field in qa and qa[a_field]:
                answer_text = qa[a_field].strip()
                break

        # Skip empty questions or answers
        if not question or not answer_text:
            continue

        # Include reasoning steps if available
        reasoning_steps = qa.get("reasoning_steps", [])
        reasoning_text = ""
        if isinstance(reasoning_steps, list):
            reasoning_text = "\n".join([
                step.get("step", "").strip()
                for step in reasoning_steps
                if isinstance(step, dict) and step.get("step", "").strip()
            ])

        # Build output (reasoning process + answer)
        if reasoning_text:
            output_text = f"{reasoning_text}\n\n{answer_text}"
        else:
            output_text = answer_text

        results.append({
            "instruction": instruction,
            "input": question,
            "output": output_text
        })

        processed_pairs += 1

    print(f"\nProcessing statistics:")
    print(f"Found QA pairs: {len(all_qa_pairs)}")
    print(f"Valid conversions: {processed_pairs}")

    if not results:
        print("No QA pairs converted!")
        return None

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as qa.json (LlamaFactory standard format)
    qa_file = output_dir / "qa.json"
    try:
        with open(qa_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        file_size = qa_file.stat().st_size
        print(f"Conversion complete: {len(results)} QA pairs -> {qa_file} ({file_size} bytes)")

        return qa_file
    except Exception as e:
        print(f"Failed to save file: {e}")
        return None


def create_llamafactory_config(output_dir):
    """Create dataset_info.json for LlamaFactory"""
    print("Creating LlamaFactory configuration...")

    dataset_info = {
        "kb_qa": {
            "file_name": "qa.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }
    }

    config_file = output_dir / "dataset_info.json"
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)

        print(f"LlamaFactory configuration created: {config_file}")
        return config_file
    except Exception as e:
        print(f"Failed to create configuration: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Convert Step 3 QA data to LlamaFactory format")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    args = parser.parse_args()

    # Handle cache_base relative path
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path = caller_cwd / cache_path

    print("Step 3 to LlamaFactory Converter")
    print("=" * 50)
    print(f"Cache base directory: {cache_path}")

    # Find input file
    input_file = find_input_file(str(cache_path))
    if not input_file:
        print("\nTips:")
        print("1. Ensure Step 3 (Text2QA generation) has been completed")
        print("2. Check if .cache/gpu/ directory exists")
        print("3. Look for text2qa_step_step3.json file")
        exit(1)

    # Output directory
    output_dir = cache_path / ".cache" / "data"

    print(f"\nStarting conversion...")
    print(f"Input: {input_file}")
    print(f"Output directory: {output_dir}")
    print("-" * 50)

    # Convert data
    qa_file = convert_to_alpaca(input_file, output_dir)

    if qa_file:
        # Create config file
        config_file = create_llamafactory_config(output_dir)

        if config_file:
            print(f"\nConversion completed successfully!")
            print(f"Files created:")
            print(f"  - {qa_file}")
            print(f"  - {config_file}")
            print("Ready for LlamaFactory training!")
        else:
            print("Configuration file creation failed")
    else:
        print("Data conversion failed")


if __name__ == "__main__":
    main()