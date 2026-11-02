"""Stage-level checkpoints for FileStorage's whole-file formats."""

from pathlib import Path

import pandas as pd
import pytest

from dataflow.core import OperatorABC
from dataflow.pipeline import BatchedPipelineABC
from dataflow.utils.storage import FileStorage


pytestmark = pytest.mark.cpu


class CopyWithCrash(OperatorABC):
    def __init__(self, crash=False):
        super().__init__()
        self.crash = crash

    def run(self, storage, input_key="value", output_key="result"):
        frame = storage.read("dataframe").copy()
        frame[output_key] = frame[input_key] * 2
        storage.write(frame)
        if self.crash:
            raise RuntimeError("injected after write")


class WholeFilePipeline(BatchedPipelineABC):
    def __init__(self, source, root, file_type, crash=False):
        super().__init__()
        self.storage = FileStorage(
            str(source),
            cache_path=str(root / "cache"),
            cache_type=file_type,
        )
        self.copy = CopyWithCrash(crash)

    def forward(self):
        self.copy.run(self.storage.step(), input_key="value", output_key="result")


def _write_source(path: Path, file_type: str) -> None:
    frame = pd.DataFrame({"value": [1, 2, 3], "text": ["a", "b", "c"]})
    if file_type == "json":
        frame.to_json(path, orient="records", force_ascii=False, indent=2)
    elif file_type == "jsonl":
        frame.to_json(path, orient="records", lines=True, force_ascii=False)
    elif file_type == "csv":
        frame.to_csv(path, index=False)
    elif file_type == "parquet":
        pytest.importorskip("pyarrow")
        frame.to_parquet(path, index=False)
    elif file_type == "pickle":
        frame.to_pickle(path)
    elif file_type == "xlsx":
        pytest.importorskip("openpyxl")
        frame.to_excel(path, index=False)
    else:
        raise AssertionError(file_type)


def _read_output(root: Path, file_type: str) -> pd.DataFrame:
    path = root / "cache" / f"dataflow_cache_step_step1.{file_type}"
    return FileStorage(
        str(path), cache_path=str(root / "read-cache"), cache_type=file_type
    ).step().read("dataframe")


@pytest.mark.parametrize(
    "file_type", ["json", "jsonl", "csv", "parquet", "pickle", "xlsx"]
)
def test_whole_file_formats_resume_after_write_failure(tmp_path, file_type):
    source = tmp_path / f"source.{file_type}"
    _write_source(source, file_type)

    restart_root = tmp_path / "restart"
    interrupted = WholeFilePipeline(source, restart_root, file_type, crash=True)
    interrupted.compile()
    with pytest.raises(RuntimeError, match="injected after write"):
        interrupted.forward()
    assert (restart_root / "cache" / ".dataflow_checkpoint" / "manifest.json").exists()

    resumed = WholeFilePipeline(source, restart_root, file_type)
    resumed.compile()
    resumed.forward()

    baseline_root = tmp_path / "baseline"
    baseline = WholeFilePipeline(source, baseline_root, file_type)
    baseline.compile()
    baseline.forward()

    pd.testing.assert_frame_equal(
        _read_output(restart_root, file_type),
        _read_output(baseline_root, file_type),
        check_dtype=False,
    )
