#!/usr/bin/env python3
"""Start DataFlex-backed training from the pdf2model CLI.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def verify_dataflex_available() -> bool:
    """Return True if ``dataflex.launcher`` imports in this Python environment."""
    try:
        subprocess.run(
            [sys.executable, "-c", "import dataflex.launcher"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError:
        print(
            "❌ DataFlex not importable in this Python. Install with: pip install -e /path/to/DataFlex"
        )
        return False


def _resolve_nproc_per_node() -> str:
    explicit = os.environ.get("NPROC_PER_NODE")
    if explicit:
        return explicit
    try:
        import torch

        return str(max(torch.cuda.device_count(), 1))
    except Exception:
        return "1"


def _want_torchrun() -> bool:
    v = os.environ.get("FORCE_TORCHRUN", "1")
    if str(v).lower() in ("1", "true", "yes"):
        return True
    try:
        import torch

        return torch.cuda.device_count() > 1
    except Exception:
        return False


def _torchrun_argv() -> list[str]:
    exe = shutil.which("torchrun")
    if exe:
        return [exe]
    return [sys.executable, "-m", "torch.distributed.run"]


def _launcher_cli_overrides() -> list[str]:
    """OmegaConf-style args after yaml; merged in ``dataflex.launcher.read_args``."""
    allow_pin = os.environ.get("PDF2MODEL_DATAFLEX_ALLOW_PIN_MEMORY", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("DATAFLOW_LESS_ALLOW_PIN_MEMORY", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if allow_pin:
        return []
    # VL batches (e.g. Qwen2.5-VL) can produce tensors with overlapping storage; pin_memory() crashes.
    return ["dataloader_pin_memory=false"]


def _build_train_command(yaml_path: Path) -> list[str]:
    y = str(yaml_path)
    tail = [y, *_launcher_cli_overrides()]
    if not _want_torchrun():
        return [sys.executable, "-m", "dataflex.launcher", *tail]

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
    nproc = _resolve_nproc_per_node()
    return _torchrun_argv() + [
        f"--nnodes={os.environ.get('NNODES', '1')}",
        f"--node_rank={os.environ.get('NODE_RANK', '0')}",
        f"--nproc_per_node={nproc}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "--module",
        "dataflex.launcher",
        *tail,
    ]


def run_dataflex_train(yaml_path: Path, cwd: Path) -> bool:
    """
    Run training via DataFlex (``dataflex.launcher``) directly.

    Default env: ``FORCE_TORCHRUN=1``, ``DISABLE_VERSION_CHECK=1``.
    ``cwd`` is the pdf2model project root (paths inside yaml are relative to it).
    """
    if not yaml_path.is_file():
        print(f"❌ DataFlex train yaml not found: {yaml_path}")
        return False
    if not verify_dataflex_available():
        return False

    env = os.environ.copy()
    env.setdefault("FORCE_TORCHRUN", "1")
    env.setdefault("DISABLE_VERSION_CHECK", "1")

    cmd = _build_train_command(yaml_path)
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {cwd}")

    try:
        subprocess.run(cmd, cwd=str(cwd), env=env, check=True, stdout=sys.stdout, stderr=sys.stderr, text=True)
        print("✅ DataFlex training completed")
        return True
    except subprocess.CalledProcessError:
        print("❌ DataFlex training failed")
        return False
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return False
