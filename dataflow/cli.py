#!/usr/bin/env python3
# dataflow/cli.py
# ===============================================================
# DataFlow 命令行入口
#   dataflow -v                         查看版本并检查更新
#   dataflow init [...]                初始化脚本/配置
#   dataflow env                       查看环境
#   dataflow webui operators [opts]    启动算子/管线 UI
#   dataflow webui agent     [opts]    启动 DataFlow-Agent UI（已整合后端）
# ===============================================================

import os, argparse, requests, sys
from colorama import init as color_init, Fore, Style

from dataflow.cli_funcs import cli_env, cli_init       # 项目已有工具
from dataflow.version import __version__               # 版本号

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
    p_init_sub.add_parser("all",       help="Init all components").set_defaults(subcommand="all")
    p_init_sub.add_parser("reasoning", help="Init reasoning components").set_defaults(subcommand="reasoning")

    # --- env ---
    top.add_parser("env", help="Show environment information")

    # --- webui ---
    p_webui = top.add_parser("webui", help="Launch Gradio WebUI")
    p_webui.add_argument("-H", "--host", default="0.0.0.0", help="Bind host (default 0.0.0.0)")
    p_webui.add_argument("-P", "--port", type=int, default=7862, help="Port (default 7862)")
    p_webui.add_argument("--show-error", action="store_true", help="Show Gradio error tracebacks")

    #    webui 二级子命令：operators / agent
    w_sub = p_webui.add_subparsers(dest="ui_mode", required=False)
    w_sub.add_parser("operators", help="Launch operator / pipeline UI")
    w_sub.add_parser("agent",     help="Launch DataFlow-Agent UI (backend included)")

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

    elif args.command == "webui":
        # 默认使用 operators
        mode = args.ui_mode or "operators"
        if mode == "operators":
            from dataflow.webui import demo
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                show_error=args.show_error,
            )
        elif mode == "agent":
            from dataflow.agent.webui import app  
            import uvicorn
            uvicorn.run(app, host=args.host, port=args.port, log_level="info")
        else:
            parser.error(f"Unknown ui_mode {mode!r}")
if __name__ == "__main__":
    main()