import os
import torch
import platform
from colorama import init, Fore, Style
from dataflow import __version__
import importlib.metadata


def is_torch_cuda_available():
    """
    Check if CUDA is available for PyTorch.
    """
    return torch.cuda.is_available()

def get_env_info():
    info = {
        "`dataflow` version": __version__,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Torchvision version": torch.__version__,
    }
    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()
        info["GPU number"] = torch.cuda.device_count()
        info["GPU memory"] = f"{torch.cuda.mem_get_info()[1] / (1024**3):.2f}GB"


    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes  # type: ignore

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    try:
        import vllm

        info["vLLM version"] = vllm.__version__
    except Exception:
        pass
    
    try:
        import sglang
        info["SGLang version"] = sglang.__version__
    except Exception:
        pass

    try:
        mineru_version = importlib.metadata.version("mineru")
        info["MinerU version"] = mineru_version
    except Exception:
        pass    

    try:
        import subprocess
        # get the dir of imdlbenco package
        imdlbenco_dir = os.path.dirname(os.path.abspath(__file__))
        # move to this dir and get the git commit hash in a subprocess
        # but don't change the current working directory
        os.chdir(imdlbenco_dir)
        commit_info = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        commit_hash = commit_info.stdout.strip()
        info["Git commit"] = commit_hash
    except Exception:
        pass
    
    print(Fore.BLUE + "=" * os.get_terminal_size().columns + Style.RESET_ALL)
    print("\n" + "\n".join([f"- {key}: {value}" for key, value in info.items()]) + "\n")
    print(Fore.BLUE + "=" * os.get_terminal_size().columns + Style.RESET_ALL)

def cli_env():
    get_env_info()

if __name__ == "__main__":
    print(get_env_info())