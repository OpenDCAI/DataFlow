
import typer
from colorama import init as color_init, Fore, Style
from typing import Optional, Tuple, List
# ---------- helpers ----------
def _echo(msg: str, color: Optional[str] = None) -> None:
    if color == "red":
        print(Fore.RED + msg + Style.RESET_ALL)
    elif color == "yellow":
        print(Fore.YELLOW + msg + Style.RESET_ALL)
    elif color == "green":
        print(Fore.GREEN + msg + Style.RESET_ALL)
    elif color == "cyan":
        print(Fore.CYAN + msg + Style.RESET_ALL)
    else:
        print(msg)