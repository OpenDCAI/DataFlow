"""Dummy CPU-only operators for testing RayAcceleratedOperator.

These are intentionally trivial, deterministic, and row-independent so
they can be used in CI without GPU resources.  Prefixed with underscore
to signal internal/test-only usage.
"""
from __future__ import annotations

from dataflow.core.operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage


class DummyDoubleOp(OperatorABC):
    """Multiplies a numeric column by 2."""

    def __init__(self):
        super().__init__()

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "value",
        output_key: str = "doubled",
    ):
        df = storage.read("dataframe")
        df[output_key] = df[input_key] * 2
        storage.write(df)


class DummyIncrementOp(OperatorABC):
    """Adds 1 to a numeric column."""

    def __init__(self):
        super().__init__()

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "doubled",
        output_key: str = "incremented",
    ):
        df = storage.read("dataframe")
        df[output_key] = df[input_key] + 1
        storage.write(df)
