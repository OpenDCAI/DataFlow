import shutil
from colorama import init, Fore, Style
from pathlib import Path

def copy_file(source: Path, destination: Path):
    """
    Copy a single file with path, with checking and asking about the existence. 
    """
    if not source.is_file():
        print(f"Error: {source} is not a file.")
        return

    if destination.exists():
        print(f"Warning: {destination.name} already exists in {destination.parent}.")
        user_input = input(f"Do you want to overwrite {destination.name}? (y/n): ").strip().lower()
        if user_input != 'y':
            print(f"Skipping {destination.name}.")
            return

    shutil.copy(source, destination)
    print(f"Copied {source.name} to {destination}.")

def copy_files_without_recursion(source: Path, destination):
    """
    Copy files under a path without recursion, with checking and asking about the existence.
    """
    yes_to_all, none_to_all = False, False

    for template_file in source.iterdir():
        if template_file.is_file():
            if template_file.name == "__init__.py":
                continue  # skip __init__.py  files
            destination_file = Path(destination) / template_file.name
            if destination_file.exists():
                if none_to_all:
                    print(f'  Skipping {template_file.name}.\n')
                    continue
                if not yes_to_all:
                    # Alert , whether overwrite?
                    print(f'  {Fore.YELLOW}Warning: {template_file.name} already exists in {destination}.{Style.RESET_ALL}')
                    user_input = input(f'  Do you want to overwrite {template_file.name}? (y/n/all/none): ').strip().lower()
                    if user_input == 'all':
                        yes_to_all = True
                    elif user_input == 'none':
                        none_to_all = True
                        print(f'  Skipping {template_file.name}.\n')
                        continue
                    elif user_input != 'y':
                        print(f'  Skipping {template_file.name}.\n')
                        continue
            shutil.copy(template_file, destination_file)
            print(f'  Copied {template_file.name} to {destination}.\n')

def copy_files_recursively(source_path: Path, destination_path: Path):
    """
    Recursively copy all contents from source_path to destination_path.
    Prompts user if a file already exists in destination.
    """
    if not source_path.exists():
        print(f"{Fore.RED}Error: Source path does not exist.{Style.RESET_ALL}")
        return

    if not source_path.is_dir():
        print(f"{Fore.RED}Error: Source path is not a directory.{Style.RESET_ALL}")
        return

    yes_to_all = False
    none_to_all = False

    for item in source_path.rglob('*'):
        relative_path = item.relative_to(source_path)
        dest_item = destination_path / relative_path

        if item.is_dir():
            dest_item.mkdir(parents=True, exist_ok=True)
        else:
            if dest_item.exists():
                if yes_to_all:
                    pass  # proceed with overwrite
                elif none_to_all:
                    print(f'  Skipping {dest_item.name}.\n')
                    continue
                else:
                    print(f'  {Fore.YELLOW}Warning: {dest_item.name} already exists in {destination_path}.{Style.RESET_ALL}')
                    user_input = input(f'  Do you want to overwrite {dest_item.name}? (y/n/all/none): ').strip().lower()
                    if user_input == 'all':
                        yes_to_all = True
                    elif user_input == 'none':
                        none_to_all = True
                        print(f'  Skipping {dest_item.name}.\n')
                        continue
                    elif user_input != 'y':
                        print(f'  Skipping {dest_item.name}.\n')
                        continue

            shutil.copy2(item, dest_item)
            # give a clear output with multi line
            print(f'{Fore.GREEN}[Copied]\nFrom: {item}\nTo: {dest_item}{Style.RESET_ALL}\n')