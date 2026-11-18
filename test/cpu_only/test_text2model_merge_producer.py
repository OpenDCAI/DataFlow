from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "dataflow"
    / "cli_funcs"
    / "text2model_pipeline"
    / "merge_json_jsonl.py"
)


def _load_module():
    assert SCRIPT.exists(), "built-in merge_json_jsonl.py must be shipped with DataFlow"
    spec = importlib.util.spec_from_file_location("text2model_merge_json_jsonl", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_manifest(cache: Path) -> list[dict]:
    manifest = cache / ".cache" / "gpu" / "text_input.jsonl"
    assert manifest.exists()
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.cpu
def test_builtin_merge_producer_is_present():
    _load_module()


@pytest.mark.cpu
def test_text_mode_creates_chunker_compatible_manifest(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    input_dir = workspace / "input"
    cache = tmp_path / "cache"
    input_dir.mkdir(parents=True)
    cache.mkdir()
    monkeypatch.setenv("PWD", str(workspace))

    (input_dir / "one.json").write_text(
        json.dumps({"text": "alpha current text"}), encoding="utf-8"
    )
    (input_dir / "two.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"content": "beta current text"}),
                json.dumps({"body": "gamma current text"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert module.scan_json_files(str(input_dir), str(cache), "texts") is True

    records = _read_manifest(cache)
    assert len(records) == 3
    assert all(set(record) == {"text_path"} for record in records)

    generated = [Path(record["text_path"]) for record in records]
    assert all(path.exists() for path in generated)
    assert {path.read_text(encoding="utf-8") for path in generated} == {
        "alpha current text",
        "beta current text",
        "gamma current text",
    }
    assert all(cache / ".cache" / "gpu" / "temp_texts" in path.parents for path in generated)


@pytest.mark.cpu
def test_nested_text_fields_remain_supported(monkeypatch, tmp_path):
    module = _load_module()
    input_dir = tmp_path / "input"
    cache = tmp_path / "cache"
    input_dir.mkdir()
    cache.mkdir()
    monkeypatch.setenv("PWD", str(tmp_path))

    (input_dir / "nested.json").write_text(
        json.dumps({"content": {"summary": "nested summary content that should survive"}}),
        encoding="utf-8",
    )

    assert module.scan_json_files(str(input_dir), str(cache), "texts") is True
    records = _read_manifest(cache)
    assert len(records) == 1
    produced = Path(records[0]["text_path"])
    assert produced.read_text(encoding="utf-8") == "nested summary content that should survive"


@pytest.mark.cpu
def test_input_cache_tree_is_not_reingested(monkeypatch, tmp_path):
    module = _load_module()
    workspace = tmp_path / "workspace"
    cache = workspace
    workspace.mkdir()
    monkeypatch.setenv("PWD", str(workspace))

    (workspace / "visible.json").write_text(json.dumps({"text": "visible text"}), encoding="utf-8")
    hidden = workspace / ".cache" / "old"
    hidden.mkdir(parents=True)
    (hidden / "stale.json").write_text(json.dumps({"text": "stale text"}), encoding="utf-8")

    assert module.scan_json_files(str(workspace), str(cache), "texts") is True
    records = _read_manifest(cache)
    assert len(records) == 1
    assert Path(records[0]["text_path"]).read_text(encoding="utf-8") == "visible text"


@pytest.mark.cpu
def test_paths_mode_preserves_original_file_index(monkeypatch, tmp_path):
    module = _load_module()
    input_dir = tmp_path / "input"
    cache = tmp_path / "cache"
    input_dir.mkdir()
    cache.mkdir()
    monkeypatch.setenv("PWD", str(tmp_path))

    source = input_dir / "source.json"
    source.write_text(json.dumps({"text": "source text"}), encoding="utf-8")

    assert module.scan_json_files(str(input_dir), str(cache), "paths") is True
    records = _read_manifest(cache)
    assert records == [{"text_path": str(source.absolute())}]


@pytest.mark.cpu
def test_empty_input_fails_and_cli_exits_nonzero(monkeypatch, tmp_path):
    module = _load_module()
    input_dir = tmp_path / "empty"
    cache = tmp_path / "cache"
    input_dir.mkdir()
    cache.mkdir()
    monkeypatch.setenv("PWD", str(tmp_path))

    assert module.scan_json_files(str(input_dir), str(cache), "texts") is False
    assert not (cache / ".cache" / "gpu" / "text_input.jsonl").exists()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(input_dir), "--cache", str(cache)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
