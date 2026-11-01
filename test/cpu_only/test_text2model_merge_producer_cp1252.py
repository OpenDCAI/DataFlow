from __future__ import annotations

import json
import os
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


@pytest.mark.cpu
def test_real_producer_is_console_encoding_safe_under_cp1252(tmp_path):
    input_dir = tmp_path / "input"
    cache = tmp_path / "cache"
    input_dir.mkdir()
    cache.mkdir()

    expected_text = "cp1252-safe producer text"
    (input_dir / "current.json").write_text(
        json.dumps({"text": expected_text}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "cp1252"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(input_dir), "--cache", str(cache)],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")

    manifest = cache / ".cache" / "gpu" / "text_input.jsonl"
    assert manifest.is_file()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1

    produced = Path(records[0]["text_path"])
    assert produced.is_file()
    assert produced.read_text(encoding="utf-8") == expected_text
