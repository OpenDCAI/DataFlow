#!/usr/bin/env python3
# dataflow/cli.py
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
# ===============================================================

import os, argparse, requests, sys
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


# ---------------- CLI 主函数 ----------------
def build_arg_parser() -> argparse.ArgumentParser:
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
    p_chat.add_argument("--model", default=None, help="Model path (default: use trained model from cache)")
    p_chat.add_argument("--cache", default="./", help="Cache directory path")

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

    # --- sft (LEGACY) ---
    p_sft = top.add_parser("sft", help="PDF to SFT training pipeline (legacy)")
    p_sft.add_argument("--pdf_path", default="./pdf", help="PDF input directory path")
    p_sft.add_argument("--lf_yaml", default="train_config.yaml", help="LlamaFactory YAML config file path")
    p_sft.add_argument("--cache", default="./", help="Cache directory path")

    return parser


def main() -> None:
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

    elif args.command == "pdf2model":
        if args.pdf2model_action == "init":
            from dataflow.cli_funcs.cli_sft import cli_pdf2model_init
            cli_pdf2model_init(cache_path=args.cache)
        elif args.pdf2model_action == "train":
            from dataflow.cli_funcs.cli_sft import cli_pdf2model_train
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
        from dataflow.cli_funcs.cli_sft import cli_pdf2model_chat
        cli_pdf2model_chat(model_path=args.model, cache_path=args.cache)

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