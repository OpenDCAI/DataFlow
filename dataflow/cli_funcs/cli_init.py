import os
from pathlib import Path
from typing import Optional
from colorama import init, Fore, Style
from .paths import DataFlowPath
from .copy_funcs import copy_files_without_recursion, copy_file, copy_files_recursively
from .utils import _echo

def _copy_scripts():
    target_dir = os.getcwd()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # script_path = DataFlowPath.get_dataflow_scripts_dir()

    copy_files_recursively(DataFlowPath.get_dataflow_scripts_dir(), target_dir)

def _copy_pipelines():
    target_dir = os.getcwd()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files_recursively(DataFlowPath.get_dataflow_pipelines_dir(), target_dir)
    # Copy pipelines

def _copy_playground():
    target_dir = os.getcwd()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files_recursively(DataFlowPath.get_dataflow_playground_dir(), target_dir)

def _copy_examples():
    target_dir = os.path.join(os.getcwd(), "example_data")
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    copy_files_recursively(DataFlowPath.get_dataflow_example_dir(), target_dir) 
    
def cli_init(subcommand):
    print(f'{Fore.GREEN}Initializing in current working directory...{Style.RESET_ALL}')
    
    # base initialize that only contain default scripts
    if subcommand == "base":
        _copy_pipelines()
        _copy_examples()
        _copy_playground()
    # if subcommand == "model_zoo":
    #     _copy_train_scripts()
    #     _copy_demo_runs() 
    #     _copy_demo_configs()
    #     _copy_dataset_json()
    # # base initialize that only contain default scripts
    # if subcommand == "backbone":
    #     _copy_train_scripts()
    #     _copy_demo_runs() 
    #     _copy_demo_configs()
    #     _copy_dataset_json()
    # print(f'{Fore.GREEN}Successfully initialized IMDLBenCo scripts.{Style.RESET_ALL}')


def init_repo_scaffold(
    no_input: bool = False,
    context: Optional[dict] = None,
) -> None:
    """
    Initialize a DataFlow repository using the built-in scaffold template.
    """
    try:
        from cookiecutter.main import cookiecutter
    except ImportError:
        raise RuntimeError(
            "cookiecutter is not installed. "
            "Please run: pip install cookiecutter"
        )

    from .paths import DataFlowPath

    template_path = DataFlowPath.get_dataflow_scaffold_dir()
    output_dir = Path.cwd()
    context = context or {}

    if not template_path.exists():
        raise FileNotFoundError(f"Scaffold template not found: {template_path}")

    _echo(f"Using scaffold template: {template_path}", "cyan")
    _echo(f"Output directory        : {output_dir}", "cyan")

    cookiecutter(
        template=str(template_path),
        no_input=no_input,
        extra_context=context,
        output_dir=str(output_dir),
    )

    _echo("Repository scaffold initialized successfully 🎉", "green")
