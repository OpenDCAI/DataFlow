#!/usr/bin/env python3
"""
DataFlow PDF2Model CLI Module - dataflow/cli_funcs/cli_pdf.py
PDF to Model training pipeline with init/train/chat commands
"""

import subprocess
import sys
import yaml
import re
import json
import os
import datetime
from pathlib import Path
from colorama import Fore, Style
from dataflow import get_logger
from .paths import DataFlowPath
from transformers import AutoConfig

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

        pdf2model_supportive_kbc_path = dataflow_path / "statics" / "pipelines" / "gpu_pipelines" / "kbcleaning" / script_name
        if pdf2model_supportive_kbc_path.exists():
            return pdf2model_supportive_kbc_path
        
        pdf2model_supportive_vqa_path = dataflow_path / "statics" / "pipelines" / "api_pipelines" / script_name
        if pdf2model_supportive_vqa_path.exists():
            return pdf2model_supportive_vqa_path

        # 检查其他可能的路径
        possible_dirs = [
            dataflow_path / "templates" / "pdf2model_pipeline",
            dataflow_path / "pipeline_templates",
            dataflow_path / "cli_funcs" / "pdf2model_pipeline",
            "./"
        ]

        for dir_path in possible_dirs:
            script_path = dir_path / script_name
            if script_path.exists():
                return script_path

        return None
    except Exception:
        return None


def copy_customizable_scripts(qa_type: str) -> bool:
    """Only copy scripts that users might want to customize"""
    print("Step 0: Copying customizable pipeline script...")

    current_dir = Path(os.getcwd())
    target_name = "pdf_to_model_pipeline.py"

    try:
        if qa_type == "vqa":
            scripts_to_copy = ["pdf_vqa_extract_pipeline.py"]
        else:  # 默认是kbc
            scripts_to_copy = ["kbcleaning_pipeline_vllm.py"]

        import shutil
        copied_files = []

        for script_name in scripts_to_copy:
            source_path = get_dataflow_script_path(script_name)
            if source_path is None:
                print(f"Warning: Template not found: {script_name}")
                continue

            target_path = current_dir / target_name

            shutil.copy2(source_path, target_path)
            copied_files.append(script_name)
            print(f"Copied: {script_name}")

        if copied_files:
            print(f"Successfully copied {len(copied_files)} customizable script(s)")
            return True
        else:
            print("No customizable scripts were copied")
            return False

    except Exception as e:
        print(f"Failed to copy scripts: {e}")
        return False


def create_train_config_yaml(cache_path="./", model_name_or_path="Qwen/Qwen2.5-7B-Instruct", qa_type="kbc"):
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
        config["template"] = infer_llama_factory_template(model_name_or_path)
        if qa_type == "vqa":
            dataset_name = "pdf_vqa_dataset"
        else:
            dataset_name = "pdf_kbc_dataset"
        config["dataset"] = dataset_name

        pdf2model_state = {
            "qa": qa_type,
            "model": model_name_or_path,
            "timestamp": timestamp
        }
        with open(cache_dir / "pdf2model_state.json", 'w') as f:
            json.dump(pdf2model_state, f, indent=2)

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

def generate_dataset_info(cache_path_obj, dataset_name, qa_type):
    """Create dataset_info.json configuration automatically"""
    dataset_info_path = cache_path_obj / ".cache" / "data" / "dataset_info.json"
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)

    if qa_type == "vqa":
        config_entry = {
            "file_name": "qa.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages", 
                "images": "images"       
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant"
            }            
        }
    else:
        config_entry = {
            "file_name": "qa.json",
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output"
            }
        }

    dataset_info = {dataset_name: config_entry}

    try:
        with open(dataset_info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"✅ Dataset info registered as '{dataset_name}' with {qa_type} format.")
        return True
    except Exception as e:
        print(f"❌ Failed to write dataset_info.json: {e}")
        return False

# 注册templates从这里获取 python -c "from llamafactory.data.template import TEMPLATES; print(list(TEMPLATES.keys()))"
def infer_llama_factory_template(model_name_or_path: str) -> str:
    """Infer llama factory model chat template through model name or path automatically"""
    SPECIAL_MAP = {
        "llama_3_2_vision": "mllama",
        "llama_3_1_vision": "mllama",
        "qwen2_5_vl": "qwen2_vl",
        "qwen2_5_omni": "qwen2_omni",
        "qwen_2": "qwen",
        "chatglm_3": "chatglm3",
        "chatglm_2": "chatglm2",
        "codegeex_4": "codegeex4",
        "deepseek_v3": "deepseek3",
        "deepseek_r1": "deepseekr1",
        "deepseek_coder": "deepseekcoder",
        "glm_z1": "glmz1",
        "internvl_2": "intern_vl",
        "internlm_2": "intern2",
        "paligemma_2": "paligemma",
        "paligemma_chat": "paligemma_chat",
    }

    SUPPORTED = [
        'alpaca', 'aquila', 'atom', 'baichuan', 'baichuan2', 'bailing', 'belle', 'bluelm', 'breeze', 'chatglm2', 'chatglm3', 
        'chatml', 'chatml_de', 'codegeex2', 'codegeex4', 'cohere', 'cpm', 'cpm3', 'cpm4', 'dbrx', 'deepseek', 'deepseek3', 
        'deepseekr1', 'deepseekcoder', 'default', 'empty', 'exaone', 'falcon', 'fewshot', 'gemma', 'gemma3', 'glm4', 'glmz1', 
        'granite3', 'granite3_vision', 'index', 'hunyuan', 'intern', 'intern2', 'intern_vl', 'kimi_vl', 'llama2', 'llama2_zh', 
        'llama3', 'llama4', 'mllama', 'moonlight', 'llava', 'llava_next', 'llava_next_llama3', 'llava_next_mistral', 
        'llava_next_qwen', 'llava_next_yi', 'llava_next_video', 'llava_next_video_mistral', 'llava_next_video_yi', 'marco', 
        'mimo', 'mimo_vl', 'minicpm_v', 'minicpm_o', 'ministral', 'mistral', 'mistral_small', 'olmo', 'openchat', 'openchat-3.6', 
        'opencoder', 'orion', 'paligemma', 'paligemma_chat', 'phi', 'phi_small', 'phi4', 'pixtral', 'qwen', 'qwen3', 'qwen2_audio', 
        'qwen2_omni', 'qwen2_vl', 'sailor', 'seed_coder', 'skywork_o1', 'smollm', 'smollm2', 'solar', 'starchat', 'telechat', 
        'telechat2', 'vicuna', 'video_llava', 'xuanyuan', 'xverse', 'yayi', 'yi', 'yi_vl', 'yuan', 'zephyr', 'ziya'
    ]

    name = model_name_or_path.split('/')[-1].lower().replace("-", "_").replace(".", "_")

    for key, template in SPECIAL_MAP.items():
        if key in name:
            return template

    VARIANTS = ['vl', 'omni', 'audio', 'v', 'video', 'r1', 'coder']
    found_in_name = [v for v in VARIANTS if f"_{v}" in f"_{name}" or f"{v}_" in f"{name}_"]

    candidates = []
    for t in SUPPORTED:
        base_word = t.split('_')[0]
        if re.search(r'\b' + re.escape(base_word) + r'\b', name.replace('_', ' ')):
            candidates.append(t)

    if not candidates:
        return "default"

    if found_in_name:
        for v in found_in_name:
            v_matches = [c for c in candidates if v in c]
            if v_matches:
                return max(v_matches, key=len)

    pure_matches = [c for c in candidates if not any(v in c for v in VARIANTS)]
    if pure_matches:
        return max(pure_matches, key=len)

    return max(candidates, key=len)

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
    customizable_script = current_dir / "pdf_to_model_pipeline.py"
    if customizable_script.exists():
        print("✅ Found customizable script: pdf_to_model_pipeline.py")
    else:
        print("❌ Missing customizable script: pdf_to_model_pipeline.py")
        print("Run 'dataflow pdf2model init' first")
        return False

    return True


def cli_pdf2model_init(cache_path: str = "./", model_name: str = "Qwen/Qwen2.5-7B-Instruct", qa_type: str = "kbc") -> bool:
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
        if not copy_customizable_scripts(qa_type):
            return False

        # Step 1: Create training configuration
        print("Step 1: Creating training configuration...")
        config_file = create_train_config_yaml(cache_path, model_name, qa_type)

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

    try:
        with open(config_path_obj, 'r', encoding='utf-8') as f:
            train_config = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ERROR] Could not read train_config.yaml: {e}")
        return False

    pdf2model_state_path = cache_path_obj / ".cache" / "pdf2model_state.json"
    try:
        with open(pdf2model_state_path, 'r', encoding='utf-8') as f:
            pdf2model_state = json.load(f)
            print(f"Loaded pipeline state from {pdf2model_state_path}")
    except Exception as e:
            print(f"Warning: State file not found at {pdf2model_state_path}, using defaults.")
            pdf2model_state = {}

    qa_type = pdf2model_state.get("qa")
    # pdf2model_model_path = pdf2model_state.get("model")

    print("-" * 60)

    try:
        # Step 1: PDF Detection
        script1_path = get_dataflow_script_path("path_to_jsonl_script.py")
        args1 = ["./", "--output", str(cache_path_obj / ".cache" / "pdf_list.jsonl"), "--qa_type", str(qa_type)]
        if not run_script_with_args(script1_path, "Step 1: PDF Detection", args1, cwd=str(current_dir)):
            return False

        # Step 2: Data Processing
        target_script_name = "pdf_to_model_pipeline.py"
        script2 = current_dir / target_script_name
        args2 = ["--cache", cache_path]
        if not run_script_with_args(script2, "Step 2: Data Processing", args2, cwd=str(current_dir)):
            return False

        # Step 2.5: Create dataset_info.json (dynamically)
        print(f"\n{Fore.BLUE}Step 2.5: Creating dataset_info.json{Style.RESET_ALL}")

        # 获取数据集名称
        dataset_name = train_config.get('dataset')
        if isinstance(dataset_name, list):
            dataset_name = dataset_name[0]  # 如果是列表，取第一个
            
        if not dataset_name:
            print("Warning: No dataset name found in train_config.yaml, using default 'kb_qa'")
            dataset_name = 'kb_qa'
            
        print(f"Dataset name from config: {dataset_name}")

        if not generate_dataset_info(cache_path_obj, dataset_name, qa_type):
            return False

        print(f"{Fore.GREEN}✅ Step 2.5: Creating dataset_info.json completed{Style.RESET_ALL}")

        # Step 3: Data Conversion - skip
        print(f"\n{Fore.BLUE}Step 3: Data Conversion{Style.RESET_ALL}")
        qa_json_path = cache_path_obj / ".cache" / "data" / "qa.json"
        if qa_json_path.exists():
            print(f"✅ qa.json already in correct format, skipping conversion")
            print(f"{Fore.GREEN}✅ Step 3: Data Conversion completed{Style.RESET_ALL}")
        else:
            print(f"❌ qa.json not found at {qa_json_path}")
            return False

        # Step 4: Training
        script4_path = get_dataflow_script_path("llama_factory_trainer.py")
        args4 = ["--config", str(config_path_obj), "--cache", cache_path]
        if not run_script_with_args(script4_path, "Step 4: Training", args4, cwd=str(current_dir)):
            return False

        # Show completion info
        try:
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                actual_output_dir = config.get('output_dir', 'unknown')
        except Exception:
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