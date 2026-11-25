from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from dataflow.cli import app
from dataflow.cli_funcs import cli_text


pytestmark = pytest.mark.cpu
runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]
TEXT2QA_PIPELINE = (
    ROOT
    / "dataflow"
    / "cli_funcs"
    / "text2model_pipeline"
    / "text_to_qa_pipeline.py"
)


def _write_train_config(cache: Path) -> Path:
    config = cache / ".cache" / "train_config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("model_name_or_path: stub\n", encoding="utf-8")
    return config


def _write_text_input(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "current.json").write_text(
        json.dumps({"text": "current selected-cache text"}),
        encoding="utf-8",
    )


def _manifest_records(cache: Path) -> list[dict]:
    manifest = cache / ".cache" / "gpu" / "text_input.jsonl"
    assert manifest.is_file()
    return [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_text2model_train_custom_cache_crosses_real_helper_boundary(monkeypatch, tmp_path):
    """A: public Typer CLI must enter the real helper and route the selected cache."""
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    selected_cache = tmp_path / "selected-cache"
    workspace.mkdir()
    _write_text_input(input_dir)
    _write_train_config(selected_cache)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))

    body_reached = {"value": False}
    calls = []

    def controlled_verify_environment() -> bool:
        body_reached["value"] = True
        return True

    def stop_after_helper_routing(
        script_path: Path,
        description: str,
        args: list = None,
        cwd: str = None,
    ) -> bool:
        calls.append((script_path, description, list(args or []), cwd))
        return False

    monkeypatch.setattr(cli_text, "verify_environment", controlled_verify_environment)
    monkeypatch.setattr(cli_text, "run_script_with_args", stop_after_helper_routing)

    result = runner.invoke(
        app,
        ["text2model", "train", str(input_dir), "--cache", str(selected_cache)],
    )

    assert "No such option" not in result.output
    assert body_reached["value"] is True
    assert calls
    assert calls[0][1] == "JSON/JSONL merging"
    assert calls[0][2] == [str(input_dir), "--cache", str(selected_cache)]
    assert calls[0][3] == str(workspace)
    # Controlled Step-1 failure proves the public wrapper still propagates
    # the real helper's False result to a nonzero process status.
    assert result.exit_code != 0


def test_public_text2model_train_default_paths_remain_compatible(monkeypatch, tmp_path):
    """A control: default cache behavior still reaches the real helper."""
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    workspace.mkdir()
    _write_text_input(input_dir)
    _write_train_config(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))

    calls = []

    def controlled_verify_environment() -> bool:
        return True

    def stop_after_helper_routing(
        script_path: Path,
        description: str,
        args: list = None,
        cwd: str = None,
    ) -> bool:
        calls.append((script_path, description, list(args or []), cwd))
        return False

    monkeypatch.setattr(cli_text, "verify_environment", controlled_verify_environment)
    monkeypatch.setattr(cli_text, "run_script_with_args", stop_after_helper_routing)

    result = runner.invoke(app, ["text2model", "train", str(input_dir)])

    assert calls
    assert calls[0][1] == "JSON/JSONL merging"
    assert calls[0][2] == [str(input_dir), "--cache", str(workspace)]
    assert calls[0][3] == str(workspace)
    assert result.exit_code != 0


def test_real_helper_default_path_executes_real_builtin_merge_producer(monkeypatch, tmp_path):
    """B: real helper must cross built-in lookup + real subprocess for Step 1."""
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    workspace.mkdir()
    _write_text_input(input_dir)
    config = _write_train_config(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))
    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)

    # Do not mock get_dataflow_script_path or run_script_with_args: Step 1 must
    # cross the real built-in lookup and real subprocess boundary. The helper is
    # allowed to fail later at the absent workspace Text2QA script.
    ok = cli_text.cli_text2model_train(
        lf_yaml=str(config),
        input_dir=str(input_dir),
    )

    records = _manifest_records(workspace)
    assert len(records) == 1
    produced_text = Path(records[0]["text_path"])
    assert produced_text.is_file()
    assert produced_text.read_text(encoding="utf-8") == "current selected-cache text"
    assert ok is False


def test_real_helper_selected_cache_executes_real_builtin_merge_producer(monkeypatch, tmp_path):
    """B: selected cache must receive real Step-1 output, not workspace cache."""
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    selected_cache = tmp_path / "selected-cache"
    workspace.mkdir()
    _write_text_input(input_dir)
    config = _write_train_config(selected_cache)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))
    monkeypatch.setattr(cli_text, "verify_environment", lambda: True)

    ok = cli_text.cli_text2model_train(
        lf_yaml=str(config),
        input_dir=str(input_dir),
        cache_path=str(selected_cache),
    )

    records = _manifest_records(selected_cache)
    assert len(records) == 1
    produced_text = Path(records[0]["text_path"])
    assert produced_text.is_file()
    assert produced_text.read_text(encoding="utf-8") == "current selected-cache text"
    assert not (workspace / ".cache" / "gpu" / "text_input.jsonl").exists()
    assert ok is False


class _StorageCapture:
    def __init__(
        self,
        first_entry_file_name: str,
        cache_path: str,
        file_name_prefix: str,
        cache_type: str,
    ):
        self.first_entry_file_name = first_entry_file_name
        self.cache_path = cache_path
        self.file_name_prefix = file_name_prefix
        self.cache_type = cache_type

    def step(self):
        return self


class _ChunkerCapture:
    def __init__(self, split_method: str, chunk_size: int, tokenizer_name: str):
        self.split_method = split_method
        self.chunk_size = chunk_size
        self.tokenizer_name = tokenizer_name


class _QAExtractorCapture:
    def __init__(self, output_json_file: str = None):
        self.output_json_file = output_json_file


class _TextCleanerCapture:
    def __init__(self, llm_serving, lang: str):
        self.llm_serving = llm_serving
        self.lang = lang


class _QAGeneratorCapture:
    def __init__(self, llm_serving, lang: str):
        self.llm_serving = llm_serving
        self.lang = lang


class _LocalServingCapture:
    def __init__(
        self,
        hf_model_name_or_path: str,
        vllm_max_tokens: int,
        vllm_tensor_parallel_size: int,
        vllm_gpu_memory_utilization: float,
        vllm_repetition_penalty: float,
    ):
        self.hf_model_name_or_path = hf_model_name_or_path
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_repetition_penalty = vllm_repetition_penalty


def _module(name: str, attrs: dict) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def test_text2qa_real_constructor_routes_selected_cache_to_qaextractor(monkeypatch, tmp_path):
    """C: execute the real Text2QAPipeline constructor and observe runtime bindings."""
    selected_cache = tmp_path / "selected-cache"
    selected_cache.mkdir()

    knowledge_cleaning = _module(
        "dataflow.operators.knowledge_cleaning",
        {
            "KBCChunkGeneratorBatch": _ChunkerCapture,
            "KBCTextCleanerBatch": _TextCleanerCapture,
            "KBCMultiHopQAGeneratorBatch": _QAGeneratorCapture,
            "QAExtractor": _QAExtractorCapture,
        },
    )
    storage = _module("dataflow.utils.storage", {"FileStorage": _StorageCapture})
    serving = _module("dataflow.serving", {"LocalModelLLMServing_vllm": _LocalServingCapture})

    monkeypatch.setitem(sys.modules, "dataflow.operators.knowledge_cleaning", knowledge_cleaning)
    monkeypatch.setitem(sys.modules, "dataflow.utils.storage", storage)
    monkeypatch.setitem(sys.modules, "dataflow.serving", serving)

    namespace = runpy.run_path(str(TEXT2QA_PIPELINE), run_name="g2t_text2qa_boundary")
    pipeline_cls = namespace["Text2QAPipeline"]
    pipeline = pipeline_cls(cache_base=str(selected_cache))

    assert pipeline.storage.first_entry_file_name == str(
        selected_cache / ".cache" / "gpu" / "text_input.jsonl"
    )
    assert pipeline.storage.cache_path == str(selected_cache / ".cache" / "gpu")
    assert pipeline.extract_format_qa.output_json_file == str(
        selected_cache / ".cache" / "data" / "qa.json"
    )
