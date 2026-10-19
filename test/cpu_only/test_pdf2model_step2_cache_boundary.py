from __future__ import annotations

import json
import runpy
import shutil
import sys
import types
from pathlib import Path

import pytest
import yaml

from dataflow.cli_funcs import cli_pdf


pytestmark = pytest.mark.cpu

ROOT = Path(__file__).resolve().parents[2]
KBC_TEMPLATE = (
    ROOT
    / "dataflow"
    / "statics"
    / "pipelines"
    / "gpu_pipelines"
    / "kbcleaning"
    / "kbcleaning_pipeline_vllm.py"
)
VQA_TEMPLATE = (
    ROOT
    / "dataflow"
    / "statics"
    / "pipelines"
    / "api_pipelines"
    / "pdf_vqa_extract_pipeline.py"
)


def _module(name: str, attrs: dict[str, object]) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_explicit_dependency_stubs(monkeypatch, capture: dict[str, object]) -> None:
    class FileStorage:
        def __init__(
            self,
            first_entry_file_name: str,
            cache_path: str,
            file_name_prefix: str,
            cache_type: str,
        ) -> None:
            resolved_manifest = Path(first_entry_file_name)
            if not resolved_manifest.is_absolute():
                resolved_manifest = (Path.cwd() / resolved_manifest).resolve()
            capture["storage"] = {
                "first_entry_file_name": first_entry_file_name,
                "resolved_manifest": str(resolved_manifest),
                "manifest_text": (
                    resolved_manifest.read_text(encoding="utf-8")
                    if resolved_manifest.is_file()
                    else None
                ),
                "cache_path": cache_path,
                "file_name_prefix": file_name_prefix,
                "cache_type": cache_type,
            }

        def step(self):
            return self

    class FileOrURLToMarkdownConverterFlash:
        def __init__(
            self,
            intermediate_dir: str,
            mineru_model_path: str,
            batch_size: int,
            replicas: int,
            num_gpus_per_replica: float,
            engine_gpu_util_rate_to_ray_cap: float,
        ) -> None:
            capture["flash_intermediate_dir"] = intermediate_dir

        def run(self, storage, input_key: str | None = None, output_key: str | None = None):
            return None

    class FileOrURLToMarkdownConverterAPI:
        def __init__(self, intermediate_dir: str) -> None:
            self.intermediate_dir = intermediate_dir

        def run(self, storage, input_key: str | None = None, output_key: str | None = None):
            return None

    class KBCChunkGenerator:
        def __init__(self, split_method: str, chunk_size: int, tokenizer_name: str) -> None:
            self.split_method = split_method

        def run(self, storage, input_key: str | None = None, output_key: str | None = None):
            return None

    class KBCTextCleaner:
        def __init__(self, llm_serving, lang: str) -> None:
            self.lang = lang

        def run(self, storage, input_key: str | None = None, output_key: str | None = None):
            return None

    class QAExtractor:
        def __init__(self, output_json_file: str) -> None:
            self.output_json_file = output_json_file
            capture["qa_output_file"] = output_json_file

        def run(
            self,
            storage,
            input_qa_key: str,
            output_instruction_key: str,
            output_question_key: str,
            output_answer_key: str,
        ):
            output = Path(self.output_json_file)
            if not output.is_absolute():
                output = Path.cwd() / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("[]", encoding="utf-8")
            return None

    class Text2MultiHopQAGenerator:
        def __init__(self, llm_serving, lang: str, num_q: int) -> None:
            self.lang = lang
            self.num_q = num_q

        def run(self, storage, input_key: str | None = None, output_key: str | None = None):
            return None

    class LocalModelLLMServingVLLM:
        def __init__(
            self,
            hf_model_name_or_path: str,
            vllm_max_tokens: int,
            vllm_tensor_parallel_size: int,
            vllm_gpu_memory_utilization: float,
            vllm_repetition_penalty: float,
        ) -> None:
            self.hf_model_name_or_path = hf_model_name_or_path

    class APILLMServingRequest:
        def __init__(
            self,
            api_url: str,
            key_name_of_api_key: str,
            model_name: str,
            max_workers: int,
        ) -> None:
            self.model_name = model_name

    class PipelineABC:
        def __init__(self) -> None:
            pass

        def compile(self):
            capture["compiled"] = True
            return None

    class QAExtractPrompt:
        def build_prompt(self) -> str:
            return "prompt"

    class PDFMerger:
        def __init__(self, output_dir: str) -> None:
            capture["pdf_merger_output_dir"] = output_dir

        def run(
            self,
            storage,
            input_pdf_list_key: str,
            input_name_key: str,
            output_pdf_path_key: str,
        ):
            return None

    class MinerU2LLMInputOperator:
        def __init__(self) -> None:
            pass

        def run(
            self,
            storage,
            input_markdown_path_key: str,
            output_converted_layout_key: str,
        ):
            return None

    class ChunkedPromptedGenerator:
        def __init__(self, llm_serving, system_prompt: str, max_chunk_len: int) -> None:
            self.system_prompt = system_prompt

        def run(self, storage, input_path_key: str, output_path_key: str):
            return None

    class LLMOutputParser:
        def __init__(self, output_dir: str, intermediate_dir: str) -> None:
            capture["llm_output_dir"] = output_dir
            capture["llm_intermediate_dir"] = intermediate_dir

        def run(
            self,
            storage,
            input_response_path_key: str,
            input_converted_layout_path_key: str,
            input_name_key: str,
            output_qalist_path_key: str,
        ):
            return None

    class QAMerger:
        def __init__(self, output_dir: str, strict_title_match: bool) -> None:
            capture["qa_merger_output_dir"] = output_dir

        def run(
            self,
            storage,
            input_qalist_path_key: str,
            input_name_key: str,
            output_merged_qalist_path_key: str,
            output_merged_md_path_key: str,
            output_qa_item_key: str,
        ):
            return None

    class VQAFormatter:
        def __init__(self, output_json_file: str) -> None:
            self.output_json_file = output_json_file
            capture["vqa_output_file"] = output_json_file

        def run(
            self,
            storage,
            input_qa_item_key: str,
            output_messages_key: str,
            output_images_key: str,
        ):
            output = Path(self.output_json_file)
            if not output.is_absolute():
                output = Path.cwd() / output
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("[]", encoding="utf-8")
            return None

    class PdfWriter:
        def __init__(self) -> None:
            pass

    modules = {
        "dataflow.operators.knowledge_cleaning": {
            "KBCChunkGenerator": KBCChunkGenerator,
            "FileOrURLToMarkdownConverterFlash": FileOrURLToMarkdownConverterFlash,
            "FileOrURLToMarkdownConverterAPI": FileOrURLToMarkdownConverterAPI,
            "KBCTextCleaner": KBCTextCleaner,
            "QAExtractor": QAExtractor,
        },
        "dataflow.operators.core_text": {
            "Text2MultiHopQAGenerator": Text2MultiHopQAGenerator,
            "ChunkedPromptedGenerator": ChunkedPromptedGenerator,
        },
        "dataflow.utils.storage": {"FileStorage": FileStorage},
        "dataflow.serving": {
            "LocalModelLLMServing_vllm": LocalModelLLMServingVLLM,
            "APILLMServing_request": APILLMServingRequest,
        },
        "dataflow.operators.pdf2vqa": {
            "MinerU2LLMInputOperator": MinerU2LLMInputOperator,
            "LLMOutputParser": LLMOutputParser,
            "QA_Merger": QAMerger,
            "PDF_Merger": PDFMerger,
            "VQAFormatter": VQAFormatter,
        },
        "dataflow.pipeline": {"PipelineABC": PipelineABC},
        "dataflow.prompts.pdf2vqa": {"QAExtractPrompt": QAExtractPrompt},
        "pypdf": {"PdfWriter": PdfWriter},
    }
    for name, attrs in modules.items():
        monkeypatch.setitem(sys.modules, name, _module(name, attrs))


def _run_real_template_cli(
    monkeypatch,
    template: Path,
    argv: list[str],
    capture: dict[str, object],
) -> None:
    _install_explicit_dependency_stubs(monkeypatch, capture)
    monkeypatch.setattr(sys, "argv", [str(template), *argv])
    runpy.run_path(str(template), run_name="__main__")


def _selected_manifest(cache: Path, marker: str) -> Path:
    manifest = cache / ".cache" / "pdf_list.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(marker, encoding="utf-8")
    return manifest


def test_kbc_real_cache_cli_consumes_selected_manifest_and_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    selected = tmp_path / "selected-kbc"
    wrong = tmp_path / "wrong-kbc"
    selected_manifest = _selected_manifest(selected, "SELECTED-KBC")
    _selected_manifest(wrong, "WRONG-KBC")
    capture: dict[str, object] = {}

    _run_real_template_cli(monkeypatch, KBC_TEMPLATE, ["--cache", str(selected)], capture)

    storage = capture["storage"]
    assert isinstance(storage, dict)
    assert storage["resolved_manifest"] == str(selected_manifest.resolve())
    assert storage["manifest_text"] == "SELECTED-KBC"
    assert storage["cache_path"] == str(selected / ".cache" / "gpu")
    assert capture["qa_output_file"] == str(selected / ".cache" / "data" / "qa.json")
    assert Path(str(capture["flash_intermediate_dir"])).is_relative_to(selected.resolve())
    assert str(wrong.resolve()) not in str(storage["resolved_manifest"])


def test_vqa_real_cache_cli_consumes_selected_manifest_and_cache(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    selected = tmp_path / "selected-vqa"
    wrong = tmp_path / "wrong-vqa"
    selected_manifest = _selected_manifest(selected, "SELECTED-VQA")
    _selected_manifest(wrong, "WRONG-VQA")
    capture: dict[str, object] = {}

    _run_real_template_cli(monkeypatch, VQA_TEMPLATE, ["--cache", str(selected)], capture)

    storage = capture["storage"]
    assert isinstance(storage, dict)
    assert storage["resolved_manifest"] == str(selected_manifest.resolve())
    assert storage["manifest_text"] == "SELECTED-VQA"
    assert storage["cache_path"] == str(selected / ".cache" / "gpu")
    assert capture["vqa_output_file"] == str(selected / ".cache" / "data" / "qa.json")
    for key in (
        "pdf_merger_output_dir",
        "flash_intermediate_dir",
        "llm_output_dir",
        "llm_intermediate_dir",
        "qa_merger_output_dir",
    ):
        assert Path(str(capture[key])).is_relative_to(selected.resolve())
    assert str(wrong.resolve()) not in str(storage["resolved_manifest"])


def test_kbc_without_cache_preserves_legacy_standalone_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    capture: dict[str, object] = {}

    _run_real_template_cli(monkeypatch, KBC_TEMPLATE, [], capture)

    storage = capture["storage"]
    assert isinstance(storage, dict)
    assert storage["first_entry_file_name"] == "../../example_data/KBCleaningPipeline/kbc_test.jsonl"
    assert storage["cache_path"] == "./.cache/gpu"
    assert capture["qa_output_file"] == "./.cache/data/qa.json"


def test_vqa_without_cache_preserves_legacy_standalone_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    capture: dict[str, object] = {}

    _run_real_template_cli(monkeypatch, VQA_TEMPLATE, [], capture)

    storage = capture["storage"]
    assert isinstance(storage, dict)
    assert storage["first_entry_file_name"] == "./example_data/PDF2VQAPipeline/vqa_extract_test.jsonl"
    assert storage["cache_path"] == "./cache"
    assert capture["vqa_output_file"] == "./.cache/data/qa.json"


def test_real_pdf_helper_handoff_reaches_step2_with_selected_cache(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    selected = tmp_path / "selected-cache"
    workspace.mkdir()
    selected_data = selected / ".cache" / "data"
    selected_data.mkdir(parents=True)
    train_config = selected / ".cache" / "train_config.yaml"
    train_config.write_text("dataset: pdf_kbc_dataset\n", encoding="utf-8")
    (selected / ".cache" / "pdf2model_state.json").write_text(
        json.dumps({"qa": "kbc", "pdf2model_train_backend": "base"}),
        encoding="utf-8",
    )
    (workspace / "current.pdf").write_bytes(b"%PDF-controlled")
    shutil.copy2(KBC_TEMPLATE, workspace / "pdf_to_model_pipeline.py")
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("PWD", str(workspace))

    capture: dict[str, object] = {"step4_seen": False}

    def environment_ok() -> bool:
        return True

    def required_files_ok() -> bool:
        return True

    def controlled_runner(
        script_path: Path,
        description: str,
        args: list | None = None,
        cwd: str | None = None,
    ) -> bool:
        argv = list(args or [])
        if description == "Step 1: PDF Detection":
            previous_argv = sys.argv
            try:
                sys.argv = [str(script_path), *argv]
                try:
                    runpy.run_path(str(script_path), run_name="__main__")
                except SystemExit as exc:
                    return exc.code in (None, 0)
                return True
            finally:
                sys.argv = previous_argv
        if description == "Step 2: Data Processing":
            _install_explicit_dependency_stubs(monkeypatch, capture)
            previous_argv = sys.argv
            try:
                sys.argv = [str(script_path), *argv]
                runpy.run_path(str(script_path), run_name="__main__")
                return True
            finally:
                sys.argv = previous_argv
        if description == "Step 4: Training (LlamaFactory)":
            capture["step4_seen"] = True
            return False
        raise AssertionError(f"Unexpected pipeline command: {description}")

    monkeypatch.setattr(cli_pdf, "verify_environment", environment_ok)
    monkeypatch.setattr(cli_pdf, "check_required_files", required_files_ok)
    monkeypatch.setattr(cli_pdf, "run_script_with_args", controlled_runner)

    result = cli_pdf.cli_pdf2model_train(
        lf_yaml=str(train_config),
        cache_path=str(selected),
    )

    selected_manifest = selected / ".cache" / "pdf_list.jsonl"
    assert selected_manifest.is_file()
    storage = capture["storage"]
    assert isinstance(storage, dict)
    assert storage["resolved_manifest"] == str(selected_manifest.resolve())
    assert storage["manifest_text"] == selected_manifest.read_text(encoding="utf-8")
    assert capture["qa_output_file"] == str(selected / ".cache" / "data" / "qa.json")
    assert capture["step4_seen"] is True
    assert result is False
