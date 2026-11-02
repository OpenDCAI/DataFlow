"""Crash-safe batch resume for the appendable JSONL and CSV formats."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import BatchedPipelineABC, StreamBatchedPipelineABC
from dataflow.utils.pipeline_checkpoint import canonical_digest
from dataflow.utils.storage import BatchedFileStorage, StreamBatchedFileStorage


pytestmark = pytest.mark.cpu


class Scale(OperatorABC):
    def __init__(self, crash=None):
        super().__init__()
        self.crash = crash
        self.calls = 0

    def run(self, storage, input_key="value", output_key="result"):
        frame = storage.read("dataframe").copy()
        frame[output_key] = frame[input_key] * 2
        call = self.calls
        self.calls += 1
        if self.crash == (call, "before"):
            raise RuntimeError("injected before write")
        storage.write(frame)
        if self.crash == (call, "after"):
            raise RuntimeError("injected after write")


class FormatPipeline(BatchedPipelineABC):
    storage_type = BatchedFileStorage

    def __init__(self, source, root, output_type, crash=None):
        super().__init__()
        self.storage = self.storage_type(
            str(source),
            cache_path=str(root / "cache"),
            cache_type=output_type,
        )
        self.scale = Scale(crash)

    def forward(self):
        self.scale.run(self.storage.step(), input_key="value", output_key="result")


class StreamFormatPipeline(FormatPipeline):
    storage_type = StreamBatchedFileStorage
    _compiled_forward = StreamBatchedPipelineABC._compiled_forward


def _write_source(path: Path, file_type: str) -> None:
    frame = pd.DataFrame(
        {
            "value": [0, 1, 2, 3, 4, 5, 6],
            "text": ["零", "一", "二", "三", "四", "五", "六"],
        }
    )
    if file_type == "jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_csv(path, index=False)


def _read_output(root: Path, file_type: str) -> pd.DataFrame:
    path = root / "cache" / f"dataflow_cache_step_step1.{file_type}"
    if file_type == "jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


@pytest.mark.parametrize("pipeline_type", [FormatPipeline, StreamFormatPipeline])
@pytest.mark.parametrize("source_type", ["jsonl", "csv"])
@pytest.mark.parametrize("output_type", ["jsonl", "csv"])
def test_default_resume_is_transactional_for_jsonl_and_csv(
    tmp_path, pipeline_type, source_type, output_type
):
    source = tmp_path / f"source.{source_type}"
    _write_source(source, source_type)

    restart_root = tmp_path / "restart"
    interrupted = pipeline_type(source, restart_root, output_type, crash=(1, "after"))
    interrupted.compile()
    with pytest.raises(RuntimeError, match="injected after write"):
        interrupted.forward(batch_size=2)

    resumed = pipeline_type(source, restart_root, output_type)
    resumed.compile()
    resumed.forward(batch_size=2)

    baseline_root = tmp_path / "baseline"
    baseline = pipeline_type(source, baseline_root, output_type)
    baseline.compile()
    baseline.forward(batch_size=2)

    pd.testing.assert_frame_equal(
        _read_output(restart_root, output_type),
        _read_output(baseline_root, output_type),
        check_dtype=False,
    )
    assert (restart_root / "cache" / ".dataflow_checkpoint" / "manifest.json").exists()

    # A second resume is idempotent and must not append the last batch again.
    resumed_again = pipeline_type(source, restart_root, output_type)
    resumed_again.compile()
    resumed_again.forward(batch_size=2)
    pd.testing.assert_frame_equal(
        _read_output(restart_root, output_type),
        _read_output(baseline_root, output_type),
        check_dtype=False,
    )


def test_manifest_created_by_583_is_migrated(tmp_path):
    source = tmp_path / "source.jsonl"
    _write_source(source, "jsonl")
    root = tmp_path / "run"
    pipeline = FormatPipeline(source, root, "jsonl")
    pipeline.compile()
    pipeline.forward(batch_size=2)

    manifest_path = root / "cache" / ".dataflow_checkpoint" / "manifest.json"
    state = json.loads(manifest_path.read_text(encoding="utf-8"))
    node = next(node for node in pipeline.op_nodes_list if node.op_obj is not None)
    source_path = source.resolve()
    legacy_payload = {
        "source": str(source_path),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "batch_size": 2,
        "config": {"compatibility": "resume_from_last"},
        "operators": [
            {
                "type": (
                    type(node.op_obj).__module__
                    + "."
                    + type(node.op_obj).__name__
                ),
                "args": node.kwargs,
            }
        ],
        "outputs": [str((root / "cache" / "dataflow_cache_step_step1.jsonl").resolve())],
        "read_policy": {"type": "NoneType", "errors_path": None},
    }
    legacy_identity = canonical_digest(legacy_payload)
    assert legacy_identity != state["identity"]
    state["identity"] = legacy_identity
    manifest_path.write_text(json.dumps(state), encoding="utf-8")

    resumed = FormatPipeline(source, root, "jsonl")
    resumed.compile()
    resumed.forward(batch_size=2)
    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert migrated["identity"] != legacy_identity
