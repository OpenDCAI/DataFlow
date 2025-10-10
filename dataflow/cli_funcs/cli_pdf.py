#!/usr/bin/env python3
"""
DataFlow PDF2Model CLI Module - dataflow/cli_funcs/cli_pdf.py
PDF to Model training pipeline with init/train/chat commands
"""

import subprocess
import sys
import yaml
import json
import os
import datetime
from pathlib import Path
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

        # PDF2Model 脚本在 dataflow/cli_funcs/pdf2model_pipeline/ 目录下
        pdf2model_path = dataflow_path / "cli_funcs" / "pdf2model_pipeline" / script_name
        if pdf2model_path.exists():
            return pdf2model_path

        # 检查其他可能的路径
        possible_dirs = [
            dataflow_path / "templates" / "pdf2model_pipeline",
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
    """Only copy scripts that users might want to customize"""
    print("Step 0: Copying customizable pipeline script...")

    current_dir = Path(os.getcwd())

    try:
        # 只复制用户可能需要自定义的脚本
        scripts_to_copy = [
            "pdf_to_qa_pipeline.py"  # 用户可能需要修改 vLLM/sglang 配置
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
            print("You can now modify these files (e.g., switch vLLM/sglang in pdf_to_qa_pipeline.py)")
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
    model_dir_name = f"pdf2model_cache_{timestamp}"  # 改为pdf2model_cache前缀

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

    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        return False

    return True


def check_required_files():
    """Check if required built-in scripts exist"""
    # 检查所有需要的内置脚本
    required_scripts = [
        "path_to_jsonl_script.py",
        "merge_filter_qa_pairs.py",
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
    customizable_script = current_dir / "pdf_to_qa_pipeline.py"
    if customizable_script.exists():
        print("✅ Found customizable script: pdf_to_qa_pipeline.py")
    else:
        print("❌ Missing customizable script: pdf_to_qa_pipeline.py")
        print("Run 'dataflow pdf2model init' first")
        return False

    return True


def cli_pdf2model_init(cache_path: str = "./", model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> bool:
    """
    PDF2Model initialization:
    0. Copy only customizable scripts to current directory
    1. Create train_config.yaml in .cache directory
    """
    print("Starting PDF2Model initialization...")
    print(f"Cache directory: {cache_path}")
    print(f"Model: {model_name}")
    print(f"Output directory: pdf2model_cache_<timestamp>")  # 更新输出目录显示
    print("-" * 60)

    if not verify_environment():
        return False

    try:
        # Step 0: Copy only customizable scripts
        if not copy_customizable_scripts():
            return False

        # Step 1: Create training configuration
        print("Step 1: Creating training configuration...")
        config_file = create_train_config_yaml(cache_path, model_name)

        if config_file:
            print("PDF2Model initialization completed!")
            return True
        else:
            print("Failed to create training configuration")
            return False

    except Exception as e:
        print(f"Initialization failed: {e}")
        return False


def get_latest_model_dir(cache_path_obj):
    """获取最新的模型目录（基于时间戳）"""
    saves_dir = cache_path_obj / ".cache" / "saves"
    if not saves_dir.exists():
        return None

    # 查找所有 pdf2model_cache_ 开头的目录
    model_dirs = []
    for dir_path in saves_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.startswith('pdf2model_cache_'):
            # 检查是否包含正确的时间戳格式 (YYYYMMDD_HHMMSS)
            timestamp_part = dir_path.name.replace('pdf2model_cache_', '')
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


def cli_pdf2model_train(lf_yaml: str = ".cache/train_config.yaml", cache_path: str = "./") -> bool:
    """
    Start PDF2Model training using mix of built-in and user scripts
    """
    print("Starting PDF2Model training...")

    current_dir = Path(os.getcwd())

    cache_path_obj = Path(cache_path)
    if not cache_path_obj.is_absolute():
        cache_path_obj = current_dir / cache_path_obj

    config_path_obj = Path(lf_yaml)
    if not config_path_obj.is_absolute():
        config_path_obj = current_dir / config_path_obj

    if not verify_environment():
        return False

    if not check_required_files():
        return False

    if not config_path_obj.exists():
        print(f"Training config file not found: {config_path_obj}")
        print(f"{Style.BRIGHT}Run 'dataflow pdf2model init' first")
        return False

    print("-" * 60)

    try:
        # Step 1: PDF Detection - 使用内置脚本
        script1_path = get_dataflow_script_path("path_to_jsonl_script.py")
        args1 = ["./", "--output", str(cache_path_obj / ".cache" / "gpu" / "pdf_list.jsonl")]
        if not run_script_with_args(script1_path, "Step 1: PDF Detection", args1, cwd=str(current_dir)):
            return False

        # Step 2: Data Processing - 使用用户目录下的脚本
        script2 = current_dir / "pdf_to_qa_pipeline.py"
        args2 = ["--cache", cache_path]
        if not run_script_with_args(script2, "Step 2: Data Processing", args2, cwd=str(current_dir)):
            return False

        # Step 3: Data Conversion - 使用内置脚本
        script3_path = get_dataflow_script_path("merge_filter_qa_pairs.py")
        args3 = ["--cache", cache_path]
        if not run_script_with_args(script3_path, "Step 3: Data Conversion", args3, cwd=str(current_dir)):
            return False

        # Step 4: Training - 使用内置脚本
        script4_path = get_dataflow_script_path("llama_factory_trainer.py")
        args4 = ["--config", str(config_path_obj), "--cache", cache_path]
        if not run_script_with_args(script4_path, "Step 4: Training", args4, cwd=str(current_dir)):
            return False

        # 显示训练完成信息，从配置文件中读取实际的输出目录
        try:
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                actual_output_dir = config.get('output_dir', 'unknown')
        except:
            actual_output_dir = 'unknown'

        print("Training completed successfully!")
        print(f"Model saved to: {actual_output_dir}")
        print("Next steps:")
        print(f"{Style.BRIGHT}Test the trained model with 'dataflow chat'")

        return True

    except Exception as e:
        print(f"Training error: {e}")
        return False


def cli_pdf2model_chat(model_path=None, cache_path="./", base_model=None):
    """Start LlamaFactory chat interface"""
    print("Starting chat interface...")

    current_dir = Path(os.getcwd())

    # 处理cache路径
    cache_path_obj = Path(cache_path)
    if not cache_path_obj.is_absolute():
        cache_path_obj = current_dir / cache_path_obj

    # 确定模型路径
    if model_path is None:
        # 获取最新的模型目录
        latest_model_dir = get_latest_model_dir(cache_path_obj)
        if latest_model_dir:
            model_path = latest_model_dir
        else:
            print("No trained model found")
            print("Run 'dataflow pdf2model train' to train a model first")
            return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run 'dataflow pdf2model train' to train a model first")
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

    # 确定基础模型路径 - 安全的读取方式
    if base_model is None:
        base_model = None  # 先设为None

        # 尝试从训练配置中读取基础模型
        config_file = cache_path_obj / ".cache" / "train_config.yaml"
        if config_file.exists():
            try:
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

        # 如果仍然没有找到base_model，报错退出而不是使用默认值
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