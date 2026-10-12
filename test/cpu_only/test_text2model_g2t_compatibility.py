from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dataflow.cli_funcs import cli_text
from dataflow.cli_funcs.text2model_pipeline.llama_factory_trainer import LlamaFactoryTrainer


pytestmark = pytest.mark.cpu


def test_trainer_preflight_reads_selected_cache_not_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    selected_cache = tmp_path / "selected-cache"
    workspace.mkdir()
    data_dir = selected_cache / ".cache" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "qa.json").write_text(
        json.dumps([{"instruction": "i", "input": "q", "output": "a"}]),
        encoding="utf-8",
    )
    (data_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))
    monkeypatch.setitem(sys.modules, "llamafactory", ModuleType("llamafactory"))

    trainer = LlamaFactoryTrainer(
        config_path=str(selected_cache / ".cache" / "train_config.yaml"),
        cache_base=str(selected_cache),
    )

    assert trainer.check_environment() is True


def test_real_helper_old_two_positional_usage_remains_valid(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    input_dir = workspace
    config = workspace / "train.yaml"
    config.write_text("model_name_or_path: stub\n", encoding="utf-8")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))

    calls = []

    def controlled_verify_environment() -> bool:
        return True

    def stop_at_step1(
        script_path: Path,
        description: str,
        args: list = None,
        cwd: str = None,
    ) -> bool:
        calls.append((script_path, description, list(args or []), cwd))
        return False

    monkeypatch.setattr(cli_text, "verify_environment", controlled_verify_environment)
    monkeypatch.setattr(cli_text, "run_script_with_args", stop_at_step1)

    ok = cli_text.cli_text2model_train(None, "train.yaml")

    assert ok is False
    assert calls
    assert calls[0][1] == "JSON/JSONL merging"
    assert calls[0][2] == [str(input_dir), "--cache", str(workspace)]
    assert calls[0][3] == str(workspace)
