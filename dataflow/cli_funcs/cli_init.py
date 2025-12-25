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
    import os
    from pathlib import Path
    from colorama import Fore, Style

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

    # ---------- pretty header ----------
    width = 80
    try:
        width = os.get_terminal_size().columns
    except OSError:
        pass

    print(Fore.BLUE + "=" * width + Style.RESET_ALL)
    print(Fore.CYAN + "🚀 DataFlow Repository Scaffold Initialization\n" + Style.RESET_ALL)

    info = {
        "Template": template_path,
        "Output Dir": output_dir,
        "Interactive": "No" if no_input else "Yes",
        "Extra Context": "Provided" if context else "None",
    }

    for k, v in info.items():
        print(f"- {k:<13}: {v}")

    print("\n" + Fore.BLUE + "-" * width + Style.RESET_ALL)
    print("Generating project from scaffold...")
    print(Fore.BLUE + "-" * width + Style.RESET_ALL + "\n")

    # ---------- cookiecutter ----------
    generated_path = cookiecutter(
        template=str(template_path),
        no_input=no_input,
        extra_context=context,
        output_dir=str(output_dir),
    )

    generated_path = Path(generated_path).resolve()
    # ---------- footer ----------
    print("\n" + Fore.GREEN + "✔ Repository scaffold initialized successfully 🎉" + Style.RESET_ALL)
    print(Fore.GREEN + f"📁 Project path: {generated_path}" + Style.RESET_ALL)

        # ---------- post-init guidance ----------
    print("\n" + Fore.CYAN + "Next steps:" + Style.RESET_ALL)
    print(f"  1. Enter project directory:")
    print(f"     cd {generated_path}")
    print()
    print(f"  2. Install this project in editable mode (for local development):")
    print(f"     pip install -e .")
    print()
    print(f"  3. Initialize a git repository (recommended for release & distribution):")
    print(f"     git init")
    print(f"     git add .")
    print(f"     git commit -m \"Initial commit\"")
    print(Fore.BLUE + "=" * width + Style.RESET_ALL)