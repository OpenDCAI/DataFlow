import pandas as pd
import pytest

from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage
from dataflow.wrapper import BatchWrapper


pytestmark = pytest.mark.cpu


class _MultiplyOperator:
    def __init__(self):
        self.calls = 0

    def run(self, storage):
        self.calls += 1
        dataframe = storage.read().copy()
        dataframe["scaled"] = dataframe["value"] * 10
        storage.write(dataframe)


class _BatchWrapperPipeline(PipelineABC):
    def __init__(self, source_path, cache_path, *, batch_cache):
        super().__init__()
        self.storage = FileStorage(
            first_entry_file_name=str(source_path),
            cache_path=str(cache_path),
            file_name_prefix="p04_batch",
            cache_type="jsonl",
        )
        self.operator = _MultiplyOperator()
        self.batch = BatchWrapper(
            self.operator,
            batch_size=2,
            batch_cache=batch_cache,
        )

    def forward(self):
        self.batch.run(self.storage.step())


@pytest.mark.parametrize("batch_cache", [False, True])
def test_batch_wrapper_runs_real_batches_during_pipeline_compile(tmp_path, batch_cache):
    source_path = tmp_path / "source.jsonl"
    cache_path = tmp_path / "cache"
    pd.DataFrame({"value": [1, 2, 3]}).to_json(
        source_path,
        orient="records",
        lines=True,
    )

    pipeline = _BatchWrapperPipeline(
        source_path,
        cache_path,
        batch_cache=batch_cache,
    )

    pipeline.compile()

    assert pipeline.operator.calls == 2
    output = pd.read_json(cache_path / "p04_batch_step1.jsonl", lines=True)
    assert output.to_dict(orient="list") == {
        "value": [1, 2, 3],
        "scaled": [10, 20, 30],
    }

    batch_cache_files = sorted(cache_path.glob("p04_batch_step0_batch*.jsonl"))
    if batch_cache:
        assert [path.name for path in batch_cache_files] == [
            "p04_batch_step0_batch0.jsonl",
            "p04_batch_step0_batch1.jsonl",
        ]
    else:
        assert batch_cache_files == []
