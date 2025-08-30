#!/usr/bin/env python3
"""
DataFlow Text Processing CLI Module - dataflow/cli_funcs/cli_text.py
Text data processing pipeline with complete workflow
"""

import subprocess
import sys
import json
import os
import datetime
from pathlib import Path
from typing import List, Union, Any
from colorama import Fore, Style
from dataflow import get_logger
from .paths import DataFlowPath

logger = get_logger()


def run_script_with_args(script_path: Path, description: str, args: list = None, cwd: str = None) -> bool:
    """Run a Python script with arguments and real-time output"""
    print(f"\n{Fore.BLUE}{description}{Style.RESET_ALL}")
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    print(f"Running: {' '.join(cmd)}")
    if cwd:
        print(f"Working directory: {cwd}")

    try:
        result = subprocess.run(cmd, cwd=cwd, check=True,
                                stdout=sys.stdout, stderr=sys.stderr, text=True)
        print(f"{Fore.GREEN}✅ {description} completed{Style.RESET_ALL}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}❌ {description} failed{Style.RESET_ALL}")
        return False


def get_dataflow_script_path(script_name: str) -> Path:
    """Get the path of dataflow built-in scripts"""
    try:
        import dataflow
        dataflow_path = Path(dataflow.__file__).parent

        # Text2Model 脚本在 dataflow/cli_funcs/text2model_pipeline/ 目录下
        text2model_path = dataflow_path / "cli_funcs" / "text2model_pipeline" / script_name
        if text2model_path.exists():
            return text2model_path

        # 检查其他可能的路径
        possible_dirs = [
            dataflow_path / "templates" / "text2model_pipeline",
            dataflow_path / "pipeline_templates"
        ]

        for dir_path in possible_dirs:
            script_path = dir_path / script_name
            if script_path.exists():
                return script_path

        return None
    except:
        return None


def copy_customizable_scripts():
    """Copy scripts that users might want to customize"""
    print("Step 0: Copying customizable pipeline scripts...")

    current_dir = Path(os.getcwd())

    try:
        # 只复制用户可能需要自定义的脚本
        scripts_to_copy = [
            "SFTDataPipeline.py"  # 用户可能需要修改算子配置
        ]

        import shutil
        copied_files = []

        for script_name in scripts_to_copy:
            source_path = get_dataflow_script_path(script_name)
            if source_path is None:
                print(f"Warning: Template not found: {script_name}")
                continue

            target_file = current_dir / script_name

            shutil.copy2(source_path, target_file)
            copied_files.append(script_name)
            print(f"Copied: {script_name}")

        if copied_files:
            print(f"Successfully copied {len(copied_files)} customizable script(s)")
            print("You can now modify these files (e.g., adjust operators in SFTDataPipeline.py)")
            return True
        else:
            print("No customizable scripts were copied")
            return False

    except Exception as e:
        print(f"Failed to copy scripts: {e}")
        return False


def create_train_config_yaml(cache_path="./", model_name_or_path="Qwen/Qwen2.5-7B-Instruct"):
    """Create train_config.yaml file using built-in LlamaFactory configuration"""
    cache_path_obj = Path(cache_path)
    if not cache_path_obj.is_absolute():
        caller_cwd = Path(os.environ.get('PWD', os.getcwd()))
        cache_path_obj = caller_cwd / cache_path_obj

    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir_name = f"text2model_cache_{timestamp}"

    cache_dir = cache_path_obj / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_file = cache_dir / "train_config.yaml"

    try:
        # 使用内置的 LlamaFactory.py 获取默认配置
        llamafactory_script_path = get_dataflow_script_path("LlamaFactory.py")
        if llamafactory_script_path is None:
            print("Built-in LlamaFactory.py not found")
            return None

        import importlib.util
        spec = importlib.util.spec_from_file_location("llamafactory_trainer", llamafactory_script_path)
        llamafactory_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(llamafactory_module)

        # 创建trainer实例并获取默认配置
        trainer = llamafactory_module.LlamaFactoryTrainer(str(config_file), str(cache_path_obj))
        config = trainer.get_default_config()

        # 只更新必要的动态参数
        config["model_name_or_path"] = model_name_or_path
        config["output_dir"] = str(cache_path_obj / ".cache" / "saves" / model_dir_name)
        config["dataset_dir"] = str(cache_path_obj / ".cache" / "data")

        # 根据模型类型设置模板
        if "qwen" in model_name_or_path.lower():
            config["template"] = "qwen"
        elif "llama" in model_name_or_path.lower():
            config["template"] = "llama3"
        elif "chatglm" in model_name_or_path.lower():
            config["template"] = "chatglm3"
        elif "baichuan" in model_name_or_path.lower():
            config["template"] = "baichuan2"

        # 保存配置
        import yaml
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f,
                      default_flow_style=False,
                      allow_unicode=True,
                      sort_keys=False,
                      indent=2)

        print(f"train_config.yaml created: {config_file}")
        print(f"Model will be saved to: {model_dir_name}")
        return str(config_file)

    except Exception as e:
        print(f"Failed to create train_config.yaml: {e}")
        return None


def verify_environment():
    """Verify runtime environment"""
    print("Checking environment...")

    missing_deps = []

    try:
        import llamafactory
        print("✅ LlamaFactory installed")
    except ImportError:
        missing_deps.append("llamafactory[torch,metrics]")

    try:
        import yaml
        print("✅ PyYAML installed")
    except ImportError:
        missing_deps.append("pyyaml")

    try:
        from dataflow.utils.storage import FileStorage
        print("✅ DataFlow storage available")
    except ImportError:
        missing_deps.append("dataflow")

    try:
        from dataflow.operators.refine import RemoveExtraSpacesRefiner
        print("✅ DataFlow operators available")
    except ImportError:
        missing_deps.append("dataflow operators")

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return False

    return True


def check_required_files_for_training():
    """Check if required built-in scripts exist for training"""
    # 检查所有需要的内置脚本
    required_scripts = [
        "mergejson&jsonl.py",
        "LlamaFactory.py"
    ]

    missing_scripts = []
    for script in required_scripts:
        script_path = get_dataflow_script_path(script)
        if script_path is None:
            missing_scripts.append(script)
        else:
            print(f"✅ Found built-in script: {script}")

    if missing_scripts:
        print(f"❌ Missing built-in scripts: {', '.join(missing_scripts)}")
        print("These should be part of the dataflow installation")
        return False

    # 检查用户目录下是否有可自定义的脚本
    current_dir = Path(os.getcwd())
    customizable_script = current_dir / "SFTDataPipeline.py"
    if customizable_script.exists():
        print("✅ Found customizable script: SFTDataPipeline.py")
    else:
        print("❌ Missing customizable script: SFTDataPipeline.py")
        print("Run 'dataflow text2model init' first")
        return False

    return True


def analyze_input_data(input_file: str) -> dict:
    """分析输入数据的字段结构"""
    if not input_file or not Path(input_file).exists():
        return {}

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                sample_data = json.loads(first_line)
                return {
                    'available_keys': list(sample_data.keys()),
                    'has_sft_format': all(key in sample_data for key in ['instruction', 'input', 'output']),
                    'has_raw_content': 'raw_content' in sample_data
                }
    except Exception as e:
        print(f"Could not analyze input file: {e}")

    return {}


def get_latest_model_dir(cache_path_obj):
    """获取最新的模型目录（基于时间戳）"""
    saves_dir = cache_path_obj / ".cache" / "saves"
    if not saves_dir.exists():
        return None

    # 查找所有 text2model_cache_ 开头的目录
    model_dirs = []
    for dir_path in saves_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith('text2model_cache_'):
            # 检查是否包含正确的时间戳格式 (YYYYMMDD_HHMMSS)
            timestamp_part = dir_path.name.replace('text2model_cache_', '')
            if len(timestamp_part) == 15 and timestamp_part[8] == '_':
                date_part = timestamp_part[:8]
                time_part = timestamp_part[9:]
                if date_part.isdigit() and time_part.isdigit() and len(time_part) == 6:
                    model_dirs.append(dir_path)

    if not model_dirs:
        return None

    # 按名称排序（时间戳会自然排序）
    model_dirs.sort(key=lambda x: x.name, reverse=True)
    return model_dirs[0]


def cli_text2model_init(cache_path: str = "./") -> bool:
    """
    Text2Model initialization:
    0. Copy only customizable scripts to current directory
    1. Create train_config.yaml in .cache directory
    """
    print("Starting Text2Model initialization...")
    print(f"Cache directory: {cache_path}")
    print(f"Model: Qwen/Qwen2.5-7B-Instruct (default)")
    print(f"Output directory: text2model_cache_<timestamp>")
    print("-" * 60)

    if not verify_environment():
        return False

    try:
        # Step 0: Copy only customizable scripts
        if not copy_customizable_scripts():
            return False

        # Step 1: Create training configuration
        print("Step 1: Creating training configuration...")
        config_file = create_train_config_yaml(cache_path, "Qwen/Qwen2.5-7B-Instruct")

        if config_file:
            print("Text2Model initialization completed!")
            return True
        else:
            print("Failed to create training configuration")
            return False

    except Exception as e:
        print(f"Initialization failed: {e}")
        return False


def cli_text2model_train(input_keys: str = None, lf_yaml: str = "./.cache/train_config.yaml") -> bool:
    """
    Start Text2Model training using mix of built-in and user scripts
    """
    print("Starting Text2Model training...")
    if input_keys:
        print(f"Processing fields: {input_keys}")

    current_dir = Path(os.getcwd())
    config_path_obj = Path(lf_yaml)
    if not config_path_obj.is_absolute():
        config_path_obj = current_dir / config_path_obj

    if not verify_environment():
        return False

    if not check_required_files_for_training():
        return False

    if not config_path_obj.exists():
        print(f"Training config file not found: {config_path_obj}")
        print("Run 'dataflow text2model init' first")
        return False

    input_dir = "./"
    cache_path_obj = current_dir
    input_path = Path(input_dir)
    if not input_path.is_absolute():
        input_path = current_dir / input_path

    if not input_path.exists():
        print(f"Input directory not found: {input_path}")
        return False

    print("-" * 60)

    try:
        # Step 1: Merge JSON/JSONL files - 使用内置脚本
        script1_path = get_dataflow_script_path("mergejson&jsonl.py")
        args1 = [str(input_path), "-o", str(cache_path_obj / ".cache" / "pt_input.jsonl")]
        if not run_script_with_args(script1_path, "Step 1: Merging JSON/JSONL files", args1, cwd=str(current_dir)):
            return False

        # 分析合并后的数据
        merged_file = str(cache_path_obj / ".cache" / "pt_input.jsonl")
        data_info = analyze_input_data(merged_file)
        print(f"Data analysis: {data_info}")

        # Step 2: Text Processing - 使用用户目录下的脚本
        script2 = current_dir / "SFTDataPipeline.py"
        args2 = ["--input", merged_file, "--cache", str(cache_path_obj / ".cache")]

        # 添加字段处理参数
        if input_keys:
            args2.extend(["--input-keys", input_keys])
        else:
            # 根据数据分析结果提供默认建议
            if data_info.get('has_sft_format', False):
                suggested_keys = "['instruction','input','output']"
                print(f"Detected SFT format data, using: {suggested_keys}")
                args2.extend(["--input-keys", suggested_keys])
            elif not data_info.get('has_raw_content', False) and data_info.get('available_keys'):
                # 如果没有raw_content但有其他字段，使用第一个可用字段
                first_key = data_info['available_keys'][0]
                print(f"Using available field: {first_key}")
                args2.extend(["--input-keys", first_key])

        if not run_script_with_args(script2, "Step 2: Text Processing", args2, cwd=str(current_dir)):
            return False

        # Step 3: Data Conversion - 创建LlamaFactory格式的数据
        print(f"\n{Fore.BLUE}Step 3: Converting to LlamaFactory format{Style.RESET_ALL}")

        # 查找处理后的数据文件
        processed_files = list(cache_path_obj.glob(".cache/gpu/sft_dataflow_cache_step_*.jsonl"))
        if not processed_files:
            print("No processed data files found")
            return False

        # 使用最新的处理文件
        latest_processed = max(processed_files, key=lambda x: x.stat().st_mtime)
        print(f"Using processed file: {latest_processed}")

        # 创建LlamaFactory数据目录和文件
        data_dir = cache_path_obj / ".cache" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        qa_file = data_dir / "qa.json"
        dataset_info_file = data_dir / "dataset_info.json"

        # 转换数据格式
        converted_data = []
        try:
            with open(latest_processed, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        # 检查是否为SFT格式
                        if 'instruction' in item and 'output' in item:
                            converted_data.append({
                                "instruction": item.get('instruction', ''),
                                "input": item.get('input', ''),
                                "output": item.get('output', '')
                            })
                        elif 'raw_content' in item:
                            # 如果是raw_content格式，创建简单的指令对
                            content = item['raw_content'][:500] + "..." if len(item['raw_content']) > 500 else item['raw_content']
                            converted_data.append({
                                "instruction": "Please analyze the following text and provide insights:",
                                "input": content,
                                "output": "This text contains information that can be analyzed for various purposes."
                            })

            # 保存qa.json
            with open(qa_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=2)

            # 创建dataset_info.json
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

            with open(dataset_info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)

            print(f"Converted {len(converted_data)} training samples")
            print(f"Created: {qa_file}")
            print(f"Created: {dataset_info_file}")
            print(f"{Fore.GREEN}✅ Step 3 completed{Style.RESET_ALL}")

        except Exception as e:
            print(f"Data conversion failed: {e}")
            return False

        # Step 4: Training - 使用内置脚本
        script4_path = get_dataflow_script_path("LlamaFactory.py")
        args4 = ["--config", str(config_path_obj)]
        if not run_script_with_args(script4_path, "Step 4: Training", args4, cwd=str(current_dir)):
            return False

        # 显示训练完成信息，从配置文件中读取实际的输出目录
        try:
            import yaml
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                actual_output_dir = config.get('output_dir', 'unknown')
        except:
            actual_output_dir = 'unknown'

        print("Training completed successfully!")
        print(f"Model saved to: {actual_output_dir}")
        print("Next steps:")
        print("Test the trained model with 'dataflow chat'")

        return True

    except Exception as e:
        print(f"Training error: {e}")
        return False


def cli_text2model_chat(model_path=None):
    """Start LlamaFactory chat interface for text2model"""
    print("Starting chat interface...")

    current_dir = Path(os.getcwd())

    # 使用默认cache路径
    cache_path_obj = current_dir

    # 确定模型路径
    if model_path is None:
        # 获取最新的模型目录
        latest_model_dir = get_latest_model_dir(cache_path_obj)
        if latest_model_dir:
            model_path = latest_model_dir
        else:
            print("No trained model found")
            print("Run 'dataflow text2model train' to train a model first")
            return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run 'dataflow text2model train' to train a model first")
        return False

    # 使用默认基础模型
    base_model = 'Qwen/Qwen2.5-7B-Instruct'

    # 尝试从训练配置中读取基础模型
    config_file = cache_path_obj / ".cache" / "train_config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                base_model = config.get('model_name_or_path', 'Qwen/Qwen2.5-7B-Instruct')
        except:
            pass

    # 检查LlamaFactory
    try:
        import llamafactory
        print("LlamaFactory available")
    except ImportError:
        print("LlamaFactory not installed")
        print("Install with: pip install llamafactory[torch,metrics]")
        return False

    # 直接用命令行参数启动聊天
    chat_cmd = [
        "llamafactory-cli", "chat",
        "--model_name_or_path", base_model,
        "--adapter_name_or_path", str(model_path.absolute())
    ]

    print(f"Base model: {base_model}")
    print(f"Adapter path: {model_path}")
    print(f"Command: {' '.join(chat_cmd)}")
    print("-" * 60)
    print("Starting chat session...")
    print("-" * 60)

    try:
        result = subprocess.run(chat_cmd, check=True)
        print("\nChat session completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nChat failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nChat session ended by user")
        return True


def cli_text_check(output_file: str = "./.cache/sft_dataflow_cache_step_step13.jsonl") -> bool:
    """Check processing results"""
    print("Checking text processing results...")

    current_dir = Path(os.getcwd())
    output_path = current_dir / output_file

    if not output_path.exists():
        print(f"Output file not found: {output_path}")
        print("Run 'dataflow text2model train' first")
        return False

    try:
        line_count = 0
        sample_lines = []

        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line.strip():
                    line_count += 1
                    if i < 3:  # Get first 3 samples
                        try:
                            data = json.loads(line)
                            sample_lines.append(data)
                        except:
                            pass

        print(f"✅ Processing completed successfully!")
        print(f"Total processed entries: {line_count}")
        print(f"Output file: {output_path}")

        if sample_lines:
            print("\nSample entries:")
            for i, sample in enumerate(sample_lines, 1):
                keys = list(sample.keys())
                print(f"  {i}. Keys: {keys}")
                for key in ['instruction', 'input', 'output', 'raw_content']:
                    if key in sample:
                        content = str(sample[key])
                        preview = content[:100] + "..." if len(content) > 100 else content
                        print(f"     {key}: {preview}")

        return True

    except Exception as e:
        print(f"Error checking results: {e}")
        return False