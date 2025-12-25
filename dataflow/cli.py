#!/usr/bin/env python3
# dataflow/cli.py — Typer 版本（更简洁、现代）
from __future__ import annotations
import re
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import typer
import requests
from colorama import init as color_init, Fore, Style

# try import project modules; fall back gracefully for tests
from dataflow.version import __version__
from dataflow.cli_funcs import cli_env, cli_init  # existing helpers
from dataflow.cli_funcs.utils import _echo

app = typer.Typer(add_completion=True, help=f"DataFlow CLI (v{__version__})")
color_init(autoreset=True)

PYPI_API = "https://pypi.org/pypi/open-dataflow/json"
ADAPTER_FILES = {"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"}
BASE_MODEL_FILES = {"config.json", "pytorch_model.bin", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}




def check_updates() -> None:
    """Print version and try to query PyPI for newer version (best-effort)."""
    cols = 80
    try:
        cols = typer.get_terminal_size()[0]
    except Exception:
        pass
    print("=" * cols)
    _echo(f"open-dataflow version: {__version__}", "cyan")
    try:
        r = requests.get(PYPI_API, timeout=5)
        r.raise_for_status()
        remote = r.json().get("info", {}).get("version")
        _echo(f"Local version : {__version__}", None)
        _echo(f"PyPI  version : {remote}", None)
        if remote and remote != __version__:
            _echo(f"New version available: {remote} — Run 'pip install -U open-dataflow' to upgrade", "yellow")
        else:
            _echo("You are using the latest version.", "green")
    except Exception as e:
        _echo("Failed to query PyPI — network may be unavailable.", "red")
        _echo(str(e), "red")
    print("=" * cols)


def dir_has_any(path: Path, names: set) -> bool:
    return any((path / n).exists() for n in names)


def detect_models_in_dir(d: Path) -> List[Tuple[str, Path]]:
    """Detect fine-tuned (adapter) or base model in directory."""
    d = d.resolve()
    if dir_has_any(d, ADAPTER_FILES):
        return [("fine_tuned_model", d)]
    if dir_has_any(d, BASE_MODEL_FILES):
        return [("base_model", d)]
    return []


def find_latest_saved_model(cache_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Find latest model folder under {cache}/.cache/saves. Returns (path, type)."""
    saves = (cache_dir / ".cache" / "saves").resolve()
    if not saves.exists():
        return None, None

    entries = []
    ts_re = re.compile(r"(\d{8}_\d{6})")
    for p in saves.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name.startswith("text2model_cache_"):
            entries.append((p, "text2model", name.replace("text2model_cache_", "")))
        elif name.startswith("pdf2model_cache_"):
            entries.append((p, "pdf2model", name.replace("pdf2model_cache_", "")))
        else:
            m = ts_re.search(name)
            if m:
                entries.append((p, "pdf2model", m.group(1)))
            else:
                # fallback to mtime
                entries.append((p, "unknown", str(int(p.stat().st_mtime))))
    if not entries:
        return None, None
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries[0][0], entries[0][1]


# ---------- chat bridges ----------
def call_dataflow_chat(model_path: Path, model_type: Optional[str] = None) -> bool:
    """Call internal dataflow chat entrypoints (text/pdf)."""
    try:
        if model_type and "text2model" in model_type:
            from dataflow.cli_funcs.cli_text import cli_text2model_chat  # type: ignore
            return bool(cli_text2model_chat(str(model_path)))
        else:
            from dataflow.cli_funcs.cli_pdf import cli_pdf2model_chat  # type: ignore
            return bool(cli_pdf2model_chat(str(model_path)))
    except ImportError as e:
        _echo(f"dataflow chat module missing: {e}", "red")
        return False
    except Exception as e:
        _echo(f"dataflow chat failed: {e}", "red")
        return False


def call_llamafactory_chat(model_path: Path) -> bool:
    """Call external llamafactory-cli when base model present."""
    try:
        subprocess.run(["llamafactory-cli", "chat", "--model_name_or_path", str(model_path)], check=True)
        return True
    except FileNotFoundError:
        _echo("llamafactory-cli not found. Install: pip install llamafactory[torch,metrics]", "red")
        return False
    except subprocess.CalledProcessError as e:
        _echo(f"llamafactory-cli failed: {e}", "red")
        return False


def start_chat(model: Optional[Path], cache_dir: Path) -> bool:
    """Main routing for `dataflow chat`."""
    if model:
        if not model.exists():
            _echo(f"Model path not found: {model}", "red")
            return False
        _echo(f"Using specified model: {model}", "cyan")
        return call_dataflow_chat(model) if dir_has_any(model, ADAPTER_FILES) else call_llamafactory_chat(model)

    # cwd
    here = detect_models_in_dir(Path.cwd())
    if here:
        for t, p in here:
            if t == "fine_tuned_model":
                _echo(f"Found fine-tuned model in cwd: {p.name}", "green")
                return call_dataflow_chat(p, "text2model")
        for t, p in here:
            if t == "base_model":
                _echo(f"Found base model in cwd: {p.name}, launching llamafactory chat", "yellow")
                return call_llamafactory_chat(p)

    # cache
    latest, mtype = find_latest_saved_model(cache_dir)
    if latest:
        _echo(f"Found trained model in cache: {latest.name}", "green")
        if dir_has_any(latest, ADAPTER_FILES):
            return call_dataflow_chat(latest, mtype)
        else:
            _echo("Cached model has no adapter files — cannot start dataflow chat.", "red")
            return False

    # nothing found
    typer.echo()
    _echo("No model found in cwd or cache.", "yellow")
    typer.echo("Options:")
    typer.echo("  - Train first: dataflow text2model init && dataflow text2model train")
    typer.echo("  - Use an existing model: dataflow chat --model /path/to/model")
    typer.echo("  - Put model files into current dir")
    return False


# ---------- eval helpers ----------
def eval_init() -> bool:
    try:
        from dataflow.cli_funcs.cli_eval import DataFlowEvalCLI  # type: ignore
        cli = DataFlowEvalCLI()
        ok = cli.init_eval_files()
        _echo("Evaluation config initialized." if ok else "Evaluation init failed.", "green" if ok else "red")
        return bool(ok)
    except ImportError as e:
        _echo(f"Eval module missing: {e}", "red")
        return False
    except Exception as e:
        _echo(f"Eval init error: {e}", "red")
        return False


def eval_run(mode: str) -> bool:
    try:
        from dataflow.cli_funcs.cli_eval import DataFlowEvalCLI  # type: ignore
        cli = DataFlowEvalCLI()
        eval_file = f"eval_{mode}.py"
        _echo(f"Running evaluation: {eval_file}", "cyan")
        ok = cli.run_eval_file(eval_file)
        _echo(f"Evaluation {mode} {'succeeded' if ok else 'failed'}.", "green" if ok else "red")
        return bool(ok)
    except ImportError as e:
        _echo(f"Eval module missing: {e}", "red")
        return False
    except Exception as e:
        _echo(f"Eval run error: {e}", "red")
        return False


# ---------- CLI commands (Typer) ----------
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Optional[bool] = typer.Option(None, "--version", "-v", help="Show version and exit"),
    # check_updates_flag: bool = typer.Option(True, "--check-updates/--no-check-updates", help="Query PyPI for updates"),
):
    """DataFlow CLI entrypoint."""
    if version:
        _echo(f"open-dataflow {__version__}", "cyan")
        # raise typer.Exit()
    # if check_updates_flag:
        check_updates()
        raise typer.Exit()
    # if invoked without subcommand, show help
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())

# ---------- init 子命令组 ----------
init_app = typer.Typer(
    name="init",
    help="Initialize DataFlow components (repo/operator/pipeline/prompt)",
    no_args_is_help=False,
)
app.add_typer(init_app, name="init")


@init_app.callback(invoke_without_command=True)
def init_root(
    ctx: typer.Context,
    sub: str = typer.Option("base", help="Which part to init (e.g. all, reasoning, base)")
):
    """
    dataflow init [SUBCOMMAND]
    If called without a subcommand, will run the 'base' init (legacy behavior).
    """
    # 如果调用时没有子命令，执行 base init（保留兼容原来 behavior）
    if ctx.invoked_subcommand is None:
        _echo(f"Initializing (base): {sub}", "cyan")
        try:
            # cli_init 在 dataflow.cli_funcs.cli_init 中实现，保持原有逻辑
            cli_init(subcommand=sub)
        except Exception as e:
            _echo(f"Init failed: {e}", "red")
            raise typer.Exit(code=1)


@init_app.command("repo", help="Initialize a new DataFlow repository from scaffold")
def init_repo(
    no_input: bool = typer.Option(False, help="Disable interactive prompts"),
):
    """Initialize a new DataFlow repository scaffold."""
    try:
        # 从你的实现里导入函数（放在 dataflow/cli_funcs/cli_init.py）
        from dataflow.cli_funcs.cli_init import init_repo_scaffold
        init_repo_scaffold(no_input=no_input)
    except Exception as e:
        _echo(f"Failed to initialize repo: {e}", "red")
        raise typer.Exit(code=1)


# 占位：后续可以添加更多子命令
@init_app.command("operator", help="Initialize an operator scaffold (TODO)")
def init_operator():
    _echo("init operator: not implemented yet", "yellow")
    raise typer.Exit(code=0)


@init_app.command("pipeline", help="Initialize a pipeline scaffold (TODO)")
def init_pipeline():
    _echo("init pipeline: not implemented yet", "yellow")
    raise typer.Exit(code=0)


@init_app.command("prompt", help="Initialize a prompt/template scaffold (TODO)")
def init_prompt():
    _echo("init prompt: not implemented yet", "yellow")
    raise typer.Exit(code=0)

@app.command()
def env():
    """Show environment info (delegates to project helper)."""
    try:
        cli_env()
    except Exception as e:
        _echo(f"env failed: {e}", "red")
        raise typer.Exit(code=1)


@app.command()
def chat(model: Optional[Path] = typer.Option(None, exists=False, help="Model path"),
         cache: Path = typer.Option(Path("."), help="Cache directory")):
    """Start an interactive chat with a model (auto-detect)."""
    ok = start_chat(model, cache)
    if not ok:
        raise typer.Exit(code=2)


# eval group
eval_app = typer.Typer(help="Evaluation commands")
app.add_typer(eval_app, name="eval")


@eval_app.command("init")
def eval_init_cmd():
    if not eval_init():
        raise typer.Exit(code=2)


@eval_app.command("api")
def eval_api_cmd():
    if not eval_run("api"):
        raise typer.Exit(code=2)


@eval_app.command("local")
def eval_local_cmd():
    if not eval_run("local"):
        raise typer.Exit(code=2)


# pdf2model
pdf_app = typer.Typer(help="PDF->model pipeline")
app.add_typer(pdf_app, name="pdf2model")


@pdf_app.command("init")
def pdf2model_init(cache: Path = typer.Option(Path("."), help="Cache dir")):
    try:
        from dataflow.cli_funcs.cli_pdf import cli_pdf2model_init  # type: ignore
        cli_pdf2model_init(cache_path=str(cache))
    except Exception as e:
        _echo(f"pdf2model init error: {e}", "red")
        raise typer.Exit(code=1)


@pdf_app.command("train")
def pdf2model_train(cache: Path = typer.Option(Path("."), help="Cache dir"),
                    lf_yaml: Optional[Path] = typer.Option(None, help="LlamaFactory yaml")):
    try:
        from dataflow.cli_funcs.cli_pdf import cli_pdf2model_train  # type: ignore
        lf = str(lf_yaml) if lf_yaml else f"{cache}/.cache/train_config.yaml"
        cli_pdf2model_train(lf_yaml=lf, cache_path=str(cache))
    except Exception as e:
        _echo(f"pdf2model train error: {e}", "red")
        raise typer.Exit(code=1)


# text2model
text_app = typer.Typer(help="Text->model pipeline")
app.add_typer(text_app, name="text2model")


@text_app.command("init")
def text2model_init(cache: Path = typer.Option(Path("."), help="Cache dir")):
    try:
        from dataflow.cli_funcs.cli_text import cli_text2model_init  # type: ignore
        cli_text2model_init(cache_path=str(cache))
    except Exception as e:
        _echo(f"text2model init error: {e}", "red")
        raise typer.Exit(code=1)


@text_app.command("train")
def text2model_train(input_dir: Path = typer.Argument(Path("."), help="Input directory"),
                     input_keys: Optional[str] = typer.Option(None, help="Input keys"),
                     lf_yaml: Optional[Path] = typer.Option(None, help="LlamaFactory yaml")):
    try:
        from dataflow.cli_funcs.cli_text import cli_text2model_train  # type: ignore
        lf = str(lf_yaml) if lf_yaml else "./.cache/train_config.yaml"
        cli_text2model_train(input_keys=input_keys, lf_yaml=lf, input_dir=str(input_dir))
    except Exception as e:
        _echo(f"text2model train error: {e}", "red")
        raise typer.Exit(code=1)


@app.command()
def webui(mode: str = typer.Option("operators", help="operators | agent | pdf")):
    """Launch (placeholder) WebUI modes."""
    if mode in {"operators", "pdf"}:
        _echo("WebUI is under maintenance; check back later.", "yellow")
    else:
        _echo("Agent UI deprecated; see DataFlow-Agent repo.", "yellow")


if __name__ == "__main__":
    app()
