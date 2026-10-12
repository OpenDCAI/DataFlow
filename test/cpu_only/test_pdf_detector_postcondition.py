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
    / "pdf2model_pipeline"
    / "path_to_jsonl_script.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("pdf_detector_postcondition", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(input_path: Path, output: Path, qa_type: str = "kbc"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(input_path),
            "--output",
            str(output),
            "--qa_type",
            qa_type,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.cpu
def test_missing_input_exits_nonzero(tmp_path):
    missing = tmp_path / "missing"
    output = tmp_path / "cache" / "pdf_list.jsonl"

    result = _run(missing, output)

    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.cpu
def test_file_input_exits_nonzero(tmp_path):
    not_dir = tmp_path / "input.pdf"
    not_dir.write_bytes(b"%PDF-controlled")
    output = tmp_path / "cache" / "pdf_list.jsonl"

    result = _run(not_dir, output)

    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.cpu
def test_empty_directory_exits_nonzero_without_manifest(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output = tmp_path / "cache" / "pdf_list.jsonl"

    result = _run(input_dir, output)

    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.cpu
def test_empty_directory_with_stale_manifest_still_exits_nonzero(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output = tmp_path / "cache" / "pdf_list.jsonl"
    output.parent.mkdir(parents=True)
    stale = '{"source":"/old/run.pdf"}\n'
    output.write_text(stale, encoding="utf-8")

    result = _run(input_dir, output)

    assert result.returncode != 0
    assert output.read_text(encoding="utf-8") == stale


@pytest.mark.cpu
def test_success_requires_generated_manifest(monkeypatch, tmp_path):
    module = _load_module()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "current.pdf").write_bytes(b"%PDF-controlled")
    output = tmp_path / "cache" / "pdf_list.jsonl"

    monkeypatch.setattr(
        module.PDFDetector,
        "generate_jsonl",
        lambda self, output_file=None, qa_type="kbc": str(output),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), str(input_dir), "--output", str(output), "--qa_type", "kbc"],
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code != 0
    assert not output.exists()


@pytest.mark.cpu
def test_valid_kbc_scan_exits_zero_and_writes_manifest(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf = input_dir / "current.pdf"
    pdf.write_bytes(b"%PDF-controlled")
    output = tmp_path / "cache" / "pdf_list.jsonl"

    result = _run(input_dir, output, "kbc")

    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [{"source": str(pdf.resolve())}]


@pytest.mark.cpu
def test_valid_vqa_scan_exits_zero_and_writes_manifest(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pdf = input_dir / "current.pdf"
    pdf.write_bytes(b"%PDF-controlled")
    output = tmp_path / "cache" / "pdf_list.jsonl"

    result = _run(input_dir, output, "vqa")

    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [{"input_pdf_paths": str(pdf.resolve()), "name": "vqa1"}]


@pytest.mark.cpu
def test_write_error_remains_nonzero(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "current.pdf").write_bytes(b"%PDF-controlled")
    output = tmp_path / "as-directory"
    output.mkdir()

    result = _run(input_dir, output, "kbc")

    assert result.returncode != 0
