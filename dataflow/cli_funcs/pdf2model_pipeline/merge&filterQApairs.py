#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path


def get_script_dir():
    """Get the absolute path of the script file directory"""
    return Path(__file__).parent.absolute()


def find_input_file(cache_base="./"):
    """Find input file relative to cache_base"""
    # Handle cache_base relative path
    cache_path = Path(cache_base)
    if not cache_path.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path = caller_cwd / cache_path

    print(f"Cache base directory: {cache_path}")

    # Possible paths relative to cache_path
    possible_paths = [
        cache_path / ".cache" / "gpu" / "batch_cleaning_step_step4.json",
        cache_path / "cache" / "gpu" / "batch_cleaning_step_step4.json",
        cache_path / "batch_cleaning_step_step4.json",
    ]

    print("Searching for input file...")
    for path in possible_paths:
        abs_path = path.resolve()  # Convert to absolute path
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

        # Check if enhanced_chunk_path exists
        enhanced_path = item.get('enhanced_chunk_path')
        if not enhanced_path:
            print("Skip (no enhanced_chunk_path)")
            continue

        # Convert to absolute path - relative to project root
        if not os.path.isabs(enhanced_path):
            # Get project root: from .cache/gpu/batch_cleaning_step_step4.json back to project root
            input_file_path = Path(input_file)  # .../cache_base/.cache/gpu/batch_cleaning_step_step4.json
            cache_gpu_dir = input_file_path.parent  # .../cache_base/.cache/gpu/
            cache_dir = cache_gpu_dir.parent  # .../cache_base/.cache/
            project_root = cache_dir.parent  # .../cache_base/

            # Remove './' from path beginning and join to project root
            clean_path = enhanced_path.lstrip('./')
            enhanced_path = project_root / clean_path

            print(f"   Path resolved: {enhanced_path}")
        else:
            enhanced_path = Path(enhanced_path)

        if not enhanced_path.exists():
            print(f"Skip (file not exists: {enhanced_path})")
            continue

        try:
            with open(enhanced_path, 'r', encoding='utf-8') as f:
                enhanced_data = json.load(f)

            # Fix: enhanced_data is chunk list, each chunk contains qa_pairs field
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
                # Debug: show data structure
                if isinstance(enhanced_data, list) and enhanced_data:
                    print(
                        f"   First element fields in list: {list(enhanced_data[0].keys()) if isinstance(enhanced_data[0], dict) else type(enhanced_data[0])}")
                elif isinstance(enhanced_data, dict):
                    print(f"   Available fields: {list(enhanced_data.keys())}")
                else:
                    print(f"   Data type: {type(enhanced_data)}")
                continue

        except Exception as e:
            print(f"Skip (read failed: {e})")
            continue

    return all_qa_pairs


def convert_to_alpaca(input_file, output_dir=None):
    """Convert to Alpaca format"""
    script_dir = get_script_dir()

    # If no output directory specified, use default .cache/data directory
    if output_dir is None:
        # Find cache_base from script_dir, then create .cache/data path
        output_dir = Path(output_dir) if output_dir else script_dir / ".cache" / "data"
    else:
        output_dir = Path(output_dir)

    print(f"Reading data file: {input_file}")
    print(f"Output directory: {output_dir}")

    results = []

    # Read main data file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully read main data, type: {type(data)}, length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
    except Exception as e:
        print(f"Failed to read file: {e}")
        return None

    if not isinstance(data, list):
        print("Expected data in list format")
        return None

    # Academic paper specific instruction
    instruction = (
        "Please answer the following question based on the provided academic literature. "
        "Your response should:\n"
        "1. Provide accurate information from the source material\n"
        "2. Include relevant scientific reasoning and methodology\n"
        "3. Reference specific findings, data, or conclusions when applicable\n"
        "4. Maintain academic rigor and precision in your explanation\n\n"
        "Focus on delivering factual, evidence-based answers suitable for academic research."
    )

    print("Loading QA pairs from enhanced files...")
    all_qa_pairs = load_qa_data_from_files(data, input_file)

    if not all_qa_pairs:
        print("No QA pairs found! Please check data structure")
        return None

    print(f"Total found {len(all_qa_pairs)} QA pairs")

    # Debug: show first QA pair structure
    if all_qa_pairs:
        print(f"Debug - First QA pair structure:")
        first_qa = all_qa_pairs[0]
        if isinstance(first_qa, dict):
            print(f"   Fields: {list(first_qa.keys())}")
            print(f"   Question field value: '{first_qa.get('question', 'N/A')}'")
            print(f"   Answer field value: '{first_qa.get('answer', 'N/A')}'")
        else:
            print(f"   Type: {type(first_qa)}")
            print(f"   Content: {first_qa}")

    # Process QA pairs
    processed_pairs = 0
    for qa in all_qa_pairs:
        if not isinstance(qa, dict):
            print(f"Warning: Skip non-dict QA: {type(qa)}")
            continue

        # Try different possible field names
        question = ""
        answer_text = ""

        # Possible question field names
        for q_field in ['question', 'Question', 'query', 'Query']:
            if q_field in qa and qa[q_field]:
                question = qa[q_field].strip()
                break

        # Possible answer field names
        for a_field in ['answer', 'Answer', 'response', 'Response']:
            if a_field in qa and qa[a_field]:
                answer_text = qa[a_field].strip()
                break

        # Debug: show first few QA questions and all fields
        if processed_pairs < 3:
            print(f"QA #{processed_pairs + 1}:")
            print(f"   All fields: {list(qa.keys())}")
            print(f"   Question: '{question}' (length: {len(question)})")
            print(f"   Answer: '{answer_text}' (length: {len(answer_text)})")

        # Skip empty questions or answers
        if not question or not answer_text:
            if processed_pairs < 3:
                print(f"   → Skip (question or answer empty)")
            continue

        # Merge reasoning steps
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


def create_llamafactory_config(output_dir=None):
    """Create dataset_info.json for LlamaFactory"""
    print("Creating LlamaFactory configuration...")

    # LlamaFactory dataset configuration
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
        print(f"Dataset name: kb_qa")
        return config_file
    except Exception as e:
        print(f"Failed to create configuration: {e}")
        return None


def verify_output(output_dir=None):
    """Verify output files"""
    print(f"\nVerifying output files (directory: {output_dir})...")

    qa_file = output_dir / "qa.json"
    config_file = output_dir / "dataset_info.json"

    # Check qa.json
    if qa_file.exists():
        size = qa_file.stat().st_size
        print(f"qa.json: {size} bytes")

        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            print(f"qa.json contains {len(qa_data)} samples")

            if qa_data:
                sample = qa_data[0]
                print(f"Sample fields: {list(sample.keys())}")
                print(f"Sample preview:")
                print(f"   Question: {sample.get('input', '')[:100]}...")
                print(f"   Answer: {sample.get('output', '')[:100]}...")
        except Exception as e:
            print(f"qa.json verification failed: {e}")
    else:
        print(f"qa.json not found")

    # Check dataset_info.json
    if config_file.exists():
        print(f"dataset_info.json exists")
    else:
        print(f"dataset_info.json not found")


def main():
    parser = argparse.ArgumentParser(description="QA data conversion tool")
    parser.add_argument("--cache", default="./", help="Cache directory path")
    args = parser.parse_args()

    # Handle cache_base relative path
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path = caller_cwd / cache_path

    print("QA Data Conversion Tool (Fixed Version)")
    print("=" * 50)
    print(f"Cache base directory: {cache_path}")

    # Find input file (relative to cache_path)
    input_file = find_input_file(str(cache_path))
    if not input_file:
        print("\nTips:")
        print("1. Ensure Pdf2QAPipeline.py has been run")
        print("2. Check if .cache/gpu/ directory exists")
        print("3. If file is in other location, please specify path manually")
        exit(1)

    # Output directory (use cache_path)
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
            print(f"\nData conversion completed!")
            verify_output(output_dir)
        else:
            print("Configuration file creation failed")
    else:
        print("Data conversion failed")

    print(f"\nAll paths are based on cache directory: {cache_path}")


if __name__ == "__main__":
    main()