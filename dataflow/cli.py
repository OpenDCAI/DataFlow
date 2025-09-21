#!/usr/bin/env python3
# dataflow/cli.py - Enhanced with local model judge support and eval init/run
# ===============================================================
# DataFlow 命令行入口
#   dataflow -v                         查看版本并检查更新
#   dataflow init [...]                初始化脚本/配置
#   dataflow env                       查看环境
#   dataflow webui operators [opts]    启动算子/管线 UI
#   dataflow webui agent     [opts]    启动 DataFlow-Agent UI（已整合后端）
#   dataflow pdf2model init/train      PDF to Model 训练流程
#   dataflow text2model init/train     Text to Model 训练流程
#   dataflow chat                      聊天界面
#   dataflow eval init                 初始化评估配置文件
#   dataflow eval api                  运行API模型评估
#   dataflow eval local                运行本地模型评估
#   dataflow eval list                 列出评估配置文件
# ===============================================================

import os
import argparse
import requests
import sys
import re
import yaml
import json
import subprocess
from pathlib import Path
from colorama import init as color_init, Fore, Style
from dataflow.cli_funcs import cli_env, cli_init  # 项目已有工具
from dataflow.version import __version__  # 版本号

color_init(autoreset=True)
PYPI_API_URL = "https://pypi.org/pypi/open-dataflow/json"


# ---------------- 版本检查 ----------------
def version_and_check_for_updates() -> None:
    width = os.get_terminal_size().columns
    print(Fore.BLUE + "=" * width + Style.RESET_ALL)
    print(f"open-dataflow codebase version: {__version__}")

    try:
        r = requests.get(PYPI_API_URL, timeout=5)
        r.raise_for_status()
        remote = r.json()["info"]["version"]
        print("\tChecking for updates...")
        print(f"\tLocal version : {__version__}")
        print(f"\tPyPI  version : {remote}")
        if remote != __version__:
            print(Fore.YELLOW + f"New version available: {remote}."
                                "  Run 'pip install -U open-dataflow' to upgrade."
                  + Style.RESET_ALL)
        else:
            print(Fore.GREEN + f"You are using the latest version: {__version__}" + Style.RESET_ALL)
    except requests.exceptions.RequestException as e:
        print(Fore.RED + "Failed to query PyPI – check your network." + Style.RESET_ALL)
        print("Error:", e)
    print(Fore.BLUE + "=" * width + Style.RESET_ALL)


# ---------------- 智能聊天功能 ----------------
def check_current_dir_for_model():
    """检查当前目录的模型文件，优先识别微调模型"""
    current_dir = Path.cwd()

    # 检查 LoRA 适配器文件
    adapter_files = [
        "adapter_config.json",
        "adapter_model.bin",
        "adapter_model.safetensors"
    ]

    # 检查基础模型文件
    model_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    # 优先检查adapter（微调模型）
    # 如果有adapter文件，就只返回微调模型，不管有没有基础模型文件
    if any((current_dir / f).exists() for f in adapter_files):
        return [("fine_tuned_model", current_dir)]

    # 只有在没有adapter文件时，才检查base model
    if any((current_dir / f).exists() for f in model_files):
        return [("base_model", current_dir)]

    return []


def get_latest_trained_model(cache_path="./"):
    """查找最新训练的模型，支持text2model和pdf2model，按时间戳排序"""
    current_dir = Path.cwd()
    cache_path_obj = Path(cache_path)
    if not cache_path_obj.is_absolute():
        cache_path_obj = current_dir / cache_path_obj

    saves_dir = cache_path_obj / ".cache" / "saves"
    if not saves_dir.exists():
        return None, None

    all_models = []

    for dir_path in saves_dir.iterdir():
        if not dir_path.is_dir():
            continue

        model_type = None
        timestamp = None

        # 检查text2model格式 (text2model_cache_YYYYMMDD_HHMMSS)
        if dir_path.name.startswith('text2model_cache_'):
            timestamp_part = dir_path.name.replace('text2model_cache_', '')
            if len(timestamp_part) == 15 and timestamp_part[8] == '_':
                date_part = timestamp_part[:8]
                time_part = timestamp_part[9:]
                if date_part.isdigit() and time_part.isdigit() and len(time_part) == 6:
                    model_type = 'text2model'
                    timestamp = timestamp_part

        # 检查pdf2model格式 (pdf2model_cache_YYYYMMDD_HHMMSS)
        elif dir_path.name.startswith('pdf2model_cache_'):
            timestamp_part = dir_path.name.replace('pdf2model_cache_', '')
            if len(timestamp_part) == 15 and timestamp_part[8] == '_':
                date_part = timestamp_part[:8]
                time_part = timestamp_part[9:]
                if date_part.isdigit() and time_part.isdigit() and len(time_part) == 6:
                    model_type = 'pdf2model'
                    timestamp = timestamp_part

        # 检查其他可能的模型目录
        else:
            # 尝试从目录名提取时间戳
            timestamp_match = re.search(r'(\d{8}_\d{6})', dir_path.name)
            if timestamp_match:
                model_type = 'pdf2model'  # 默认为pdf2model
                timestamp = timestamp_match.group(1)
            elif 'qwen' in dir_path.name.lower() or 'model' in dir_path.name.lower():
                # 如果找不到时间戳但看起来像模型目录，使用修改时间
                model_type = 'pdf2model'  # 默认为pdf2model
                mtime = dir_path.stat().st_mtime
                # 将修改时间转换为timestamp格式以便排序
                import datetime
                dt = datetime.datetime.fromtimestamp(mtime)
                timestamp = dt.strftime("%Y%m%d_%H%M%S")

        if model_type and timestamp:
            all_models.append((dir_path, model_type, timestamp))

    if not all_models:
        return None, None

    # 按时间戳排序，最新的在前（不管是什么类型的模型）
    all_models.sort(key=lambda x: x[2], reverse=True)
    latest_model_path, model_type, timestamp = all_models[0]

    return latest_model_path, model_type


def call_dataflow_chat(model_path, model_type=None):
    """调用dataflow的聊天功能（用于微调模型）"""
    # 判断模型类型
    if model_type is None:
        # 从路径判断类型
        path_str = str(model_path)
        if 'text2model' in path_str:
            model_type = 'text2model'
        elif 'pdf2model' in path_str:
            model_type = 'pdf2model'
        else:
            # 无法判断，默认尝试text2model
            model_type = 'text2model'

    if model_type == 'text2model':
        try:
            from dataflow.cli_funcs.cli_text import cli_text2model_chat
            return cli_text2model_chat(str(model_path))
        except ImportError:
            print("Cannot find text model chat function")
            return False
    elif model_type == 'pdf2model':
        try:
            from dataflow.cli_funcs.cli_pdf import cli_pdf2model_chat
            return cli_pdf2model_chat(str(model_path))
        except ImportError:
            print("Cannot find PDF model chat function")
            return False
    else:
        print(f"Unknown model type: {model_type}")
        return False


def call_llamafactory_chat(model_path):
    """调用llamafactory的聊天功能（用于基础模型）"""
    import subprocess

    chat_cmd = [
        "llamafactory-cli", "chat",
        "--model_name_or_path", str(model_path)
    ]

    try:
        result = subprocess.run(chat_cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"LlamaFactory chat failed: {e}")
        return False
    except FileNotFoundError:
        print("llamafactory-cli not found. Please install LlamaFactory:")
        print("pip install llamafactory[torch,metrics]")
        return False


def smart_chat_command(model_path=None, cache_path="./"):
    """智能聊天命令，统一处理各种模型类型，不自动下载"""

    if model_path:
        # 如果明确指定了模型路径，直接使用
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            print(f"Specified model path does not exist: {model_path}")
            return False

        print(f"{Fore.CYAN}Using specified model: {model_path}{Style.RESET_ALL}")

        # 检查是否有adapter文件
        adapter_files = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors"
        ]

        has_adapter = any((model_path_obj / f).exists() for f in adapter_files)

        if has_adapter:
            # 有adapter，使用dataflow chat
            return call_dataflow_chat(model_path)
        else:
            # 没有adapter，使用llamafactory chat
            return call_llamafactory_chat(model_path)

    # 检查当前目录
    detected_models = check_current_dir_for_model()

    if detected_models:
        # 优先使用fine_tuned_model（adapter）
        for model_type, path in detected_models:
            if model_type == "fine_tuned_model":
                print(f"{Fore.GREEN}Found trained model in current directory: {path.name}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Starting chat interface...{Style.RESET_ALL}")
                return call_dataflow_chat(path)

        # 如果没有adapter，使用base_model
        for model_type, path in detected_models:
            if model_type == "base_model":
                print(f"{Fore.YELLOW}Found base model in current directory: {path.name}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}Starting chat interface...{Style.RESET_ALL}")
                return call_llamafactory_chat(path)

    # 检查缓存中的训练模型
    latest_model, model_type = get_latest_trained_model(cache_path)

    if latest_model:
        model_name = Path(latest_model).name
        print(f"{Fore.GREEN}Found trained model from cache: {model_name}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Starting chat interface...{Style.RESET_ALL}")

        # 检查缓存中的模型是否有adapter文件
        latest_model_path = Path(latest_model)
        adapter_files = [
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors"
        ]

        has_adapter = any((latest_model_path / f).exists() for f in adapter_files)
        if has_adapter:
            return call_dataflow_chat(latest_model, model_type)
        else:
            print(f"No adapter files found in {latest_model}")
            print("This doesn't appear to be a trained model directory.")
            return False

    # 如果什么都没找到，给出提示而不下载
    print("No model found in current directory or cache.")
    print()
    print("Options:")
    print("1. Train a model first:")
    print("   dataflow text2model init && dataflow text2model train")
    print("   dataflow pdf2model init && dataflow pdf2model train")
    print()
    print("2. Use an existing model:")
    print("   dataflow chat --model /path/to/your/model")
    print()
    print("3. Download a model manually and place it in current directory")
    return False


# ---------------- 新的eval命令处理函数 ----------------
def handle_python_config_init(eval_type: str, output_file: str = None):
    """处理Python配置文件初始化"""
    try:
        from dataflow.cli_funcs.cli_eval import DataFlowEvalCLI
        
        cli = DataFlowEvalCLI()
        success = cli.init_eval_file(eval_type, output_file)
        
        if success:
            print("✅ 配置文件初始化成功")
        else:
            print("❌ 配置文件初始化失败")
            
        return success
        
    except ImportError as e:
        print(f"Python配置评估模块不可用：{e}")
        print("请检查 dataflow.cli_funcs.cli_eval 模块是否存在")
        return False
    except Exception as e:
        print(f"配置文件初始化失败：{e}")
        return False


def handle_python_config_eval(eval_type: str, args=None):
    """处理Python配置文件评估模式"""
    try:
        from dataflow.cli_funcs.cli_eval import DataFlowEvalCLI
        
        cli = DataFlowEvalCLI()
        
        # 使用默认文件名
        eval_file = f"eval_{eval_type}.py"
        
        print(f"🚀 开始{eval_type}模型评估：{eval_file}")
        
        # 传递命令行参数到评估器
        success = cli.run_eval_file(eval_type, eval_file, args)
        
        if success:
            print(f"✅ {eval_type}模型评估完成")
        else:
            print(f"❌ {eval_type}模型评估失败")
            
        return success
        
    except ImportError as e:
        print(f"Python配置评估模块不可用：{e}")
        print("请检查 dataflow.cli_funcs.cli_eval 模块是否存在")
        return False
    except Exception as e:
        print(f"Python配置评估失败：{e}")
        return False


def list_eval_files():
    """列出评估配置文件"""
    try:
        from dataflow.cli_funcs.cli_eval import DataFlowEvalCLI
        
        cli = DataFlowEvalCLI()
        cli.list_eval_files()
        return True
        
    except ImportError:
        print("Python配置评估模块不可用")
        return False
    except Exception as e:
        print(f"列出配置文件失败：{e}")
        return False


def handle_eval_command(args):
    """Handle evaluation command - 支持自动检测和模型指定"""
    try:
        eval_action = getattr(args, 'eval_action', None)
        
        # 处理 init 子命令
        if eval_action == 'init':
            return handle_python_config_init(args.type, args.output)
        
        # 处理 api 子命令（增强版）
        elif eval_action == 'api':
            return handle_python_config_eval('api', args)
        
        # 处理 local 子命令（增强版）
        elif eval_action == 'local':
            return handle_python_config_eval('local', args)
        
        # 处理 list 子命令
        elif eval_action == 'list':
            return list_eval_files()
        
        # 如果没有指定子命令，显示帮助
        else:
            print("DataFlow 评估工具")
            print()
            print("可用命令:")
            print("  dataflow eval init [--type api/local]     # 初始化评估配置文件")
            print("  dataflow eval api                         # 运行API模型评估（自动检测模型）")
            print("  dataflow eval local                       # 运行本地模型评估（自动检测模型）")
            print("  dataflow eval list                        # 列出配置文件")
            print()
            print("高级用法:")
            print("  dataflow eval api --models model1,model2  # 指定特定模型进行评估")
            print("  dataflow eval api --no-auto               # 禁用自动检测，使用配置文件中的模型")
            print()
            print("完整评估流程:")
            print("  1. dataflow eval api                      # 自动检测本地模型并评估")
            print("  2. 查看生成的评估报告                      # model_comparison_report.json")
            print()
            print("配置文件说明:")
            print("  - eval_api.py: API评估器配置（GPT-4o等）")
            print("  - eval_local.py: 本地评估器配置")
            return False
        
    except Exception as e:
        print(f"评估命令执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ---------------- CLI 主函数 ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    """构建参数解析器"""
    parser = argparse.ArgumentParser(
        prog="dataflow",
        description=f"DataFlow Command-Line Interface  (v{__version__})",
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version and exit")

    # ============ 顶层子命令 ============ #
    top = parser.add_subparsers(dest="command", required=False)

    # --- init ---
    p_init = top.add_parser("init", help="Initialize scripts/configs in current dir")
    p_init_sub = p_init.add_subparsers(dest="subcommand", required=False)
    p_init_sub.add_parser("all", help="Init all components").set_defaults(subcommand="all")
    p_init_sub.add_parser("reasoning", help="Init reasoning components").set_defaults(subcommand="reasoning")

    # --- env ---
    top.add_parser("env", help="Show environment information")

    # --- chat ---
    p_chat = top.add_parser("chat", help="Start chat interface with trained model")
    p_chat.add_argument("--model", default=None, help="Model path (default: use latest trained model from cache)")
    p_chat.add_argument("--cache", default="./", help="Cache directory path")

    # --- eval 命令（修改版本，支持模型参数） ---
    p_eval = top.add_parser("eval", help="Model evaluation using BenchDatasetEvaluator")
    eval_sub = p_eval.add_subparsers(dest="eval_action", help="Evaluation actions")

    # eval init 子命令
    eval_init = eval_sub.add_parser("init", help="Initialize evaluation configuration file")
    eval_init.add_argument("--type", choices=["api", "local"], default="api",
                          help="Configuration type: api (API models) or local (local models)")
    eval_init.add_argument("--output", help="Output file name (default: eval_api.py or eval_local.py)")

    # eval api 子命令（增强版）
    eval_api = eval_sub.add_parser("api", help="Run API model evaluation")
    eval_api.add_argument("--models", help="Comma-separated list of models to evaluate (overrides config)")
    eval_api.add_argument("--no-auto", action="store_true", help="Disable auto-detection of models")

    # eval local 子命令（增强版）
    eval_local = eval_sub.add_parser("local", help="Run local model evaluation")
    eval_local.add_argument("--models", help="Comma-separated list of models to evaluate (overrides config)")
    eval_local.add_argument("--no-auto", action="store_true", help="Disable auto-detection of models")

    # eval list 子命令
    eval_list = eval_sub.add_parser("list", help="List evaluation configuration files")

    # --- pdf2model ---
    p_pdf2model = top.add_parser("pdf2model", help="PDF to model training pipeline")
    p_pdf2model.add_argument("--cache", default="./", help="Cache directory path")
    p_pdf2model_sub = p_pdf2model.add_subparsers(dest="pdf2model_action", required=True)

    p_pdf2model_init = p_pdf2model_sub.add_parser("init", help="Initialize PDF to model pipeline")

    p_pdf2model_train = p_pdf2model_sub.add_parser("train", help="Start training after PDF processing")
    p_pdf2model_train.add_argument("--lf_yaml", default=None,
                                   help="LlamaFactory config file (default: {cache}/.cache/train_config.yaml)")

    # --- text2model ---
    p_text2model = top.add_parser("text2model", help="Train model from JSON/JSONL data")
    p_text2model_sub = p_text2model.add_subparsers(dest="text2model_action", required=True)

    p_text2model_init = p_text2model_sub.add_parser("init", help="Initialize text2model pipeline")
    p_text2model_init.add_argument("--cache", default="./", help="Cache directory path")

    p_text2model_train = p_text2model_sub.add_parser("train", help="Start training after text processing")
    p_text2model_train.add_argument('input_dir', nargs='?', default='./',
                                    help='Input directory to scan (default: ./)')
    p_text2model_train.add_argument('--input-keys', default=None,
                                    help='Fields to process (default: text)')
    p_text2model_train.add_argument("--lf_yaml", default=None,
                                    help="LlamaFactory config file (default: {cache}/.cache/train_config.yaml)")

    # --- webui ---
    p_webui = top.add_parser("webui", help="Launch Gradio WebUI")
    p_webui.add_argument("-H", "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    p_webui.add_argument("-P", "--port", type=int, default=7862, help="Port (default 7862)")
    p_webui.add_argument("--show-error", action="store_true", help="Show Gradio error tracebacks")

    #    webui 二级子命令：operators / agent
    w_sub = p_webui.add_subparsers(dest="ui_mode", required=False)
    w_sub.add_parser("operators", help="Launch operator / pipeline UI")
    w_sub.add_parser("agent", help="Launch DataFlow-Agent UI (backend included)")
    w_sub.add_parser("pdf", help="Launch PDF Knowledge Base Cleaning UI")

    return parser


def main() -> None:
    """主入口函数"""
    parser = build_arg_parser()
    args = parser.parse_args()

    # ---------- 顶层逻辑分发 ----------
    if args.version:
        version_and_check_for_updates()
        return

    if args.command == "init":
        cli_init(subcommand=args.subcommand or "base")

    elif args.command == "env":
        cli_env()

    elif args.command == "eval":
        handle_eval_command(args)

    elif args.command == "pdf2model":
        if args.pdf2model_action == "init":
            from dataflow.cli_funcs.cli_pdf import cli_pdf2model_init
            cli_pdf2model_init(cache_path=args.cache)
        elif args.pdf2model_action == "train":
            from dataflow.cli_funcs.cli_pdf import cli_pdf2model_train
            # If no lf_yaml specified, use default path relative to cache
            lf_yaml = args.lf_yaml or f"{args.cache}/.cache/train_config.yaml"
            cli_pdf2model_train(lf_yaml=lf_yaml, cache_path=args.cache)

    elif args.command == "text2model":
        from dataflow.cli_funcs.cli_text import cli_text2model_init, cli_text2model_train

        if args.text2model_action == "init":
            cli_text2model_init(cache_path=getattr(args, 'cache', './'))
        elif args.text2model_action == "train":
            # 如果没有指定lf_yaml，使用默认路径
            lf_yaml = getattr(args, 'lf_yaml', None) or "./.cache/train_config.yaml"
            cli_text2model_train(input_keys=getattr(args, 'input_keys', None), lf_yaml=lf_yaml)

    elif args.command == "chat":
        smart_chat_command(model_path=args.model, cache_path=args.cache)

    elif args.command == "webui":
        # 默认使用 operators
        mode = args.ui_mode or "operators"
        if mode == "operators":
            from dataflow.webui.operator_pipeline import demo
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                show_error=args.show_error,
            )
        elif mode == "agent":
            from dataflow.agent.webui import app
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        elif mode == "pdf":
            from dataflow.webui import kbclean_webui
            kbclean_webui.create_ui().launch()
        else:
            parser.error(f"Unknown ui_mode {mode!r}")


if __name__ == "__main__":
    main()