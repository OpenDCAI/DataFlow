from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from dataflow.utils.storage import DataFlowStorage


class InMemoryStorage(DataFlowStorage):
    """Lightweight in-memory ``DataFlowStorage`` for use inside Ray actors.

    Avoids filesystem I/O and step-file coupling so that each actor
    replica can independently read a DataFrame chunk and write results back.

    Typical lifecycle inside ``_OpRunner.run``::

        storage = InMemoryStorage(df_chunk)
        some_dataflow_op.run(storage, input_key="text", output_key="score")
        result = storage.result  # DataFrame written by the operator

    This storage does **not** participate in ``PipelineABC.compile()``,
    so ``step()`` simply returns ``self`` (no copy needed).
    A single ``_df`` is mutated in-place throughout the lifecycle:
    ``write()`` replaces it, ``read()`` returns it.
    """

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.operator_step = 0

    # --- DataFlowStorage ABC ---

    def read(self, output_type: Literal["dataframe", "dict"] = "dataframe") -> Any:
        if output_type == "dataframe":
            return self._df
        if output_type == "dict":
            return self._df.to_dict("records")
        raise ValueError(f"Unsupported output_type: {output_type}")

    def write(self, data: Any) -> Any:
        if isinstance(data, pd.DataFrame):
            self._df = data
        elif isinstance(data, list):
            self._df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type for write: {type(data)}")
        return None

    def get_keys_from_dataframe(self) -> list[str]:
        return self._df.columns.tolist()

    def step(self):
        self.operator_step += 1
        return self

    # --- helpers ---

    @property
    def result(self) -> pd.DataFrame:
        """Return the current DataFrame (after any writes)."""
        return self._df
