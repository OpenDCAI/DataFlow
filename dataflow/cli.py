import os
import argparse
import requests

from colorama import init, Fore, Style

# from dataflow.utils.paths import BencoPath
from dataflow.cli_funcs import cli_env, cli_init
import importlib.metadata 

PYPI_API_URL = 'https://pypi.org/pypi/open-dataflow/json'
from dataflow.version import __version__

def version_and_check_for_updates():
    # print a bar by the length of the shell width
    print(Fore.BLUE + "=" * os.get_terminal_size().columns + Style.RESET_ALL)
    print(f'open-dataflow codebase version: {__version__}')
    try:
        response = requests.get(PYPI_API_URL, timeout=5)
        response.raise_for_status()  # 如果响应码不是200，则抛出异常
        pypi_data = response.json()
        cloud_version = pypi_data['info']['version']  # 获取最新版本号
        # cloud_version = '0.1.21'   # for debug & test
        print("\tChecking for updates...")
        print("\tLocal version: ", __version__)
        print("\tPyPI newest version: ", cloud_version)

        local_version = __version__  # 通过 importlib.metadata 获取当前安装版本

        if cloud_version != local_version:
            print(Fore.YELLOW + f"New version available: {cloud_version}. Your version: {local_version}." + Style.RESET_ALL)
            print("Run 'pip install --upgrade open-dataflow' to upgrade.")           
        else:
            print(Fore.GREEN + f"You are using the latest version: {local_version}." + Style.RESET_ALL)
    except requests.exceptions.RequestException as e:
        print(Fore.RED + "Failed to check for updates from PyPI. Please check your internet connection." + Style.RESET_ALL)
        print(f"Error: {e}")
    print(Fore.BLUE + "=" * os.get_terminal_size().columns + Style.RESET_ALL)
def main():
    parser = argparse.ArgumentParser(description='Command line interface for DataFlow, with codebase version: ' + __version__)

    # Add version argument with update check only when user requests version
    parser.add_argument('-v', '--version', action='store_true', help="Show the version of the tool")
    
    subparsers = parser.add_subparsers(dest='command', required=False)
    
    # init command
    parser_init = subparsers.add_parser('init', help='Initialize the scripts and configs in a directory')
    init_subparsers = parser_init.add_subparsers(dest='subcommand', required=False)
    
    # init all
    parser_init_all = init_subparsers.add_parser('all', help='Initialize all components')
    parser_init_all.set_defaults(subcommand='all')

    # init reasoning
    parser_init_reasoning = init_subparsers.add_parser('reasoning', help='Initialize reasoning components')
    parser_init_reasoning.set_defaults(subcommand='reasoning')

    # env command
    parser_env = subparsers.add_parser('env', help='Show environment information')
    
    # parser.add_argument('--config', type=str, help='Path to the configuration file')
    
    args = parser.parse_args()
    if args.version:
        version_and_check_for_updates()
    
    if args.command == 'init':
        if args.subcommand is None:
            args.subcommand = 'base'
        cli_init(subcommand=args.subcommand)
        # print("TODO Calling cli_init with subcommand:", args.subcommand)
        from dataflow.cli_funcs.paths import DataFlowPath
        # print(DataFlowPath.get_dataflow_dir())
        # print(DataFlowPath.get_dataflow_scripts_dir())
    elif args.command == 'env':
        cli_env()

if __name__ == '__main__':
    main()
