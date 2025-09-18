#!/usr/bin/env python3
"""
DataFlow Text Processing CLI Module - dataflow/cli_funcs/cli_text.py
Text data processing pipeline with complete workflow including Text2QA
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
    print("Step 0: Setting up customizable pipeline scripts...")

    current_dir = Path(os.getcwd())

    # 检查当前目录下是否已经存在所需的脚本文件
    required_scripts = [
        "text_to_qa_pipeline.py",
    ]

    existing_scripts = []
    missing_scripts = []

    for script_name in required_scripts:
        script_path = current_dir / script_name
        if script_path.exists():
            existing_scripts.append(script_name)
            print(f"Found existing: {script_name}")
        else:
            missing_scripts.append(script_name)

    # 尝试从模板复制缺失的脚本
    copied_files = []
    for script_name in missing_scripts:
        source_path = get_dataflow_script_path(script_name)
        if source_path is not None:
            try:
                import shutil
                target_file = current_dir / script_name
                shutil.copy2(source_path, target_file)
                copied_files.append(script_name)
                print(f"Copied from template: {script_name}")
            except Exception as e:
                print(f"Warning: Failed to copy {script_name}: {e}")
        else:
            print(f"Warning: Template not found for {script_name}")

    total_available = len(existing_scripts) + len(copied_files)

    if total_available > 0:
        print(f"Setup completed: {total_available} scripts available")
        if existing_scripts:
            print(f"  Existing scripts: {', '.join(existing_scripts)}")
        if copied_files:
            print(f"  Copied from templates: {', '.join(copied_files)}")
        return True
    else:
        print("Warning: No pipeline scripts available")
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
        llamafactory_script_path = get_dataflow_script_path("llama_factory_trainer.py")
        if llamafactory_script_path is None:
            print("Built-in llama_factory_trainer.py not found")
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
        # 修复: 使用正确的算子导入路径和类名
        from dataflow.operators.knowledge_cleaning import (
            KBCChunkGeneratorBatch as CorpusTextSplitterBatch,
            KBCTextCleanerBatch as KnowledgeCleanerBatch,
            KBCMultiHopQAGeneratorBatch as MultiHopQAGeneratorBatch
        )
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
        "merge_json_jsonl.py",
        "llama_factory_trainer.py"
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
    customizable_scripts = [
        "text_to_qa_pipeline.py",
    ]

    missing_customizable = []
    for script_name in customizable_scripts:
        script_path = current_dir / script_name
        if script_path.exists():
            print(f"✅ Found customizable script: {script_name}")
        else:
            missing_customizable.append(script_name)

    if missing_customizable:
        print(f"❌ Missing customizable scripts: {', '.join(missing_customizable)}")
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
                    'has_text_field': 'text' in sample_data,
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
    0. Check for existing scripts and copy any missing templates
    1. Create train_config.yaml in .cache directory
    """

    if not verify_environment():
        return False

    try:
        # Step 0: Check for existing scripts and setup missing ones
        if not copy_customizable_scripts():
            print("Warning: Some scripts may be missing, but continuing...")

        # Step 1: Create training configuration
        print("Step 1: Creating training configuration...")
        config_file = create_train_config_yaml(cache_path, "Qwen/Qwen2.5-7B-Instruct")

        if config_file:
            print("Text2Model initialization completed!")
            print("\nWorkflow:")
            print("1. Put your JSON/JSONL files with 'text' field in current directory")
            print("2. Run: dataflow text2model train")
            print("   This will automatically run Text2QA generation and training")
            return True
        else:
            print("Failed to create training configuration")
            return False

    except Exception as e:
        print(f"Initialization failed: {e}")
        return False


def cli_text2model_train(input_keys: str = None, lf_yaml: str = "./.cache/train_config.yaml") -> bool:
    """
    Start Text2Model training using complete pipeline
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
        # Step 1: Merge JSON/JSONL files to create text_input.jsonl
        print(f"{Fore.CYAN}Step 1: Merging JSON/JSONL files...{Style.RESET_ALL}")

        # 调用 merge_json_jsonl.py 的逻辑
        script1_path = get_dataflow_script_path("merge_json_jsonl.py")
        args1 = [str(input_path), "--cache", str(cache_path_obj)]
        if not run_script_with_args(script1_path, "JSON/JSONL merging", args1, cwd=str(current_dir)):
            print(f"{Fore.RED}❌ Step 1: JSON/JSONL merging failed{Style.RESET_ALL}")
            return False

        # 验证 text_input.jsonl 是否创建成功
        text_input_file = cache_path_obj / ".cache" / "gpu" / "text_input.jsonl"
        if not text_input_file.exists():
            print(
                f"{Fore.RED}❌ text_input.jsonl not created. Check if you have JSON/JSONL files in {input_path}{Style.RESET_ALL}")
            return False

        file_size = text_input_file.stat().st_size
        print(f"{Fore.GREEN}✅ Step 1 completed: {text_input_file} ({file_size} bytes){Style.RESET_ALL}")

        # Step 2: Text2QA Pipeline
        print(f"{Fore.CYAN}Step 2: Text2QA generation...{Style.RESET_ALL}")

        script2_path = cache_path_obj / "text_to_qa_pipeline.py"
        args2 = ["--cache", str(cache_path_obj)]
        if not run_script_with_args(script2_path, "Text2QA generation", args2, cwd=str(current_dir)):
            print(f"{Fore.RED}❌ Step 2: Text2QA generation failed{Style.RESET_ALL}")
            return False

        # 验证 Text2QA 输出
        qa_output_file = cache_path_obj / ".cache" / "gpu" / "text2qa_step_step3.json"
        if not qa_output_file.exists():
            print(f"{Fore.RED}❌ Text2QA output not found{Style.RESET_ALL}")
            return False

        file_size = qa_output_file.stat().st_size
        print(f"{Fore.GREEN}✅ Step 2 completed: {qa_output_file} ({file_size} bytes){Style.RESET_ALL}")

        # Step 3: Convert to training format
        print(f"{Fore.CYAN}Step 3: Converting to training format...{Style.RESET_ALL}")

        script3_path = get_dataflow_script_path("merge_filter_qa_pairs.py")
        args3 = ["--cache", str(cache_path_obj)]
        if not run_script_with_args(script3_path, "QA format conversion", args3, cwd=str(current_dir)):
            print(f"{Fore.RED}❌ Step 3: QA format conversion failed{Style.RESET_ALL}")
            return False

        # 验证训练数据
        qa_file = cache_path_obj / ".cache" / "data" / "qa.json"
        dataset_info_file = cache_path_obj / ".cache" / "data" / "dataset_info.json"

        if not qa_file.exists() or not dataset_info_file.exists():
            print(f"{Fore.RED}❌ Training data files not created{Style.RESET_ALL}")
            return False

        # 统计样本数
        try:
            import json
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            sample_count = len(qa_data)
            file_size = qa_file.stat().st_size
            print(
                f"{Fore.GREEN}✅ Step 3 completed: {sample_count} training samples ({file_size} bytes){Style.RESET_ALL}")
        except:
            print(f"{Fore.GREEN}✅ Step 3 completed{Style.RESET_ALL}")

        # Step 4: Training
        print(f"{Fore.CYAN}Step 4: Starting model training...{Style.RESET_ALL}")

        script4_path = get_dataflow_script_path("llama_factory_trainer.py")
        args4 = ["--config", str(config_path_obj), "--cache", str(cache_path_obj)]
        if not run_script_with_args(script4_path, "Model training", args4, cwd=str(current_dir)):
            print(f"{Fore.RED}❌ Step 4: Training failed{Style.RESET_ALL}")
            return False

        print(f"{Fore.GREEN}✅ Text2Model training completed successfully!{Style.RESET_ALL}")
        print(f"Next steps:")
        print(f"  Test the model: dataflow chat")

        return True

    except Exception as e:
        print(f"Training error: {e}")
        return False


def _run_text2qa_workflow(current_dir: Path, cache_path_obj: Path, config_path_obj: Path) -> bool:
    """Run Text2QA workflow"""

    # Step 1: Check if Text2QA output exists
    text2qa_output = cache_path_obj / ".cache" / "gpu" / "text2qa_step_step3.json"

    if not text2qa_output.exists():
        print("Text2QA output not found. Please run text_to_qa_pipeline.py first.")
        print("Example:")
        print("  1. Prepare JSON/JSONL files with 'text' field in current directory")
        print("  2. Run: python text_to_qa_pipeline.py")
        print("  3. Then run: dataflow text2model train --text2qa")
        return False

    print("Found Text2QA output, proceeding with conversion...")

    # Step 2: Convert QA to Alpaca format
    script2 = current_dir / "merge_filter_qa_pairs.py"
    args2 = ["--cache", str(cache_path_obj)]

    if not run_script_with_args(script2, "Step 2: Converting QA to Alpaca format", args2, cwd=str(current_dir)):
        return False

    # Step 3: Training
    script3_path = get_dataflow_script_path("llama_factory_trainer.py")
    args3 = ["--config", str(config_path_obj)]
    if not run_script_with_args(script3_path, "Step 3: Training", args3, cwd=str(current_dir)):
        return False

    # 显示训练完成信息
    try:
        import yaml
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            actual_output_dir = config.get('output_dir', 'unknown')
    except:
        actual_output_dir = 'unknown'

    print("Text2QA training completed successfully!")
    print(f"Model saved to: {actual_output_dir}")
    print("Next steps:")
    print("Test the trained model with 'dataflow chat'")
    return True


# def _run_normal_text_workflow(input_path: Path, current_dir: Path, cache_path_obj: Path, config_path_obj: Path,
#                               input_keys: str) -> bool:
#     """Run Text2QA workflow as the main processing pipeline"""

#     # Step 1: Merge JSON/JSONL files and create text_input.jsonl - 使用内置脚本
#     script1_path = get_dataflow_script_path("merge_json_jsonl.py")
#     args1 = [str(input_path), "--cache", str(cache_path_obj)]
#     if not run_script_with_args(script1_path, "Step 1: Preparing text input for Text2QA", args1, cwd=str(current_dir)):
#         return False

#     # Step 2: Run Text2QA Pipeline - 使用用户目录下的脚本
#     script2 = current_dir / "text_to_qa_pipeline.py"
#     args2 = ["--cache", str(cache_path_obj)]

#     if not run_script_with_args(script2, "Step 2: Text2QA generation", args2, cwd=str(current_dir)):
#         return False

#     # Step 3: Convert QA to Alpaca format - 使用用户目录下的脚本
#     script3 = current_dir / "merge_filter_qa_pairs.py"
#     args3 = ["--cache", str(cache_path_obj)]

#     if not run_script_with_args(script3, "Step 3: Converting QA to training format", args3, cwd=str(current_dir)):
#         return False

#     # Step 4: Training - 使用内置脚本
#     script4_path = get_dataflow_script_path("llama_factory_trainer.py")
#     args4 = ["--config", str(config_path_obj)]
#     if not run_script_with_args(script4_path, "Step 4: Training", args4, cwd=str(current_dir)):
#         return False

#     # 显示训练完成信息，从配置文件中读取实际的输出目录
#     try:
#         import yaml
#         with open(config_path_obj, 'r', encoding='utf-8') as f:
#             config = yaml.safe_load(f)
#             actual_output_dir = config.get('output_dir', 'unknown')
#     except:
#         actual_output_dir = 'unknown'

#     print("Text2QA training completed successfully!")
#     print(f"Model saved to: {actual_output_dir}")
#     print("Next steps:")
#     print("Test the trained model with 'dataflow chat'")
#     return True


def cli_text2model_chat(model_path=None):
    """Start LlamaFactory chat interface for text2model"""

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

    # 验证是否为有效的adapter目录
    adapter_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors"
    ]

    has_adapter = any((model_path / f).exists() for f in adapter_files)
    if not has_adapter:
        print(f"No adapter files found in {model_path}")
        print("This doesn't appear to be a trained adapter directory.")
        print("Expected files: adapter_config.json, adapter_model.bin/safetensors")
        return False

    # 安全地确定基础模型
    base_model = None

    # 尝试从训练配置中读取基础模型
    config_file = cache_path_obj / ".cache" / "train_config.yaml"
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                base_model = config.get('model_name_or_path')
                if base_model:
                    print(f"Found base model in config: {base_model}")
        except Exception as e:
            print(f"Warning: Could not read config file: {e}")

    # 尝试从adapter_config.json读取
    if not base_model:
        adapter_config_path = model_path / "adapter_config.json"
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r', encoding='utf-8') as f:
                    adapter_config = json.load(f)
                    base_model = adapter_config.get('base_model_name_or_path')
                    if base_model:
                        print(f"Found base model in adapter config: {base_model}")
            except Exception as e:
                print(f"Warning: Could not read adapter config: {e}")

    # 如果仍然没有找到base_model，报错退出
    if not base_model:
        print("Cannot determine base model path")
        print("Please ensure your training config contains 'model_name_or_path'")
        print("Or check that adapter_config.json exists and contains 'base_model_name_or_path'")
        return False

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