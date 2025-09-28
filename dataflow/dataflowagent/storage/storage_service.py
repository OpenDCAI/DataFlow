#!/usr/bin/env python3
"""
storage_service.py  ── Enhanced FileStorage with sampling utilities
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-02

This module provides the `SampleFileStorage` class, which extends standard file storage
with advanced sampling capabilities, including:

* Efficient statistical and reservoir sampling from large datasets
* Built-in support for sample size estimation (Cochran formula) for both proportions and means
* Streamed and in-memory data access for pandas DataFrames
* Helper methods for counting, field listing, and full/streamed record fetching

Designed for rapid data inspection, human evaluation, and scalable sampling tasks.
Thread-safety and memory usage depend on the base FileStorage implementation and dataset size.

"""

import random
from typing import Iterator, Dict, Any, List, Tuple, Literal
import math
import pandas as pd
from dataflow.utils.storage import FileStorage
class SampleFileStorage(FileStorage):
    """
    A FileStorage that supports statistical / reservoir sampling.

    Compared with `read()`, the `rsample()` method lets you
    draw a subset of records with statistical guarantees, which is
    useful for quick data inspection or human evaluation.

    Three sampling modes are supported:

    1. manual      – fixed-size reservoir `k`
    2. proportion  – sample size for estimating a **proportion**
                    (e.g. accuracy) with margin ± *e*
    3. mean        – sample size for estimating a **mean**
                    (e.g. rating) with margin ± *e*
    """

    def _load_as_dataframe(self) -> pd.DataFrame:
        """
        Always fetch the CURRENT step as a pandas DataFrame.
        (Reuses the .read() logic in the parent class.)
        """
        return self.read("dataframe")

    def _stream_dicts(self) -> Iterator[Dict[str, Any]]:
        """Iterate row-by-row as dict without storing the whole dataset."""
        df = self._load_as_dataframe()
        for _, row in df.iterrows():
            yield row.to_dict()

    def count(self) -> int:
        """Total number of records -- utilised by the sample-size formulae."""
        return len(self._load_as_dataframe())

    def get_fields(self) -> List[str]:
        """Get column names / JSON keys in the dataset."""
        df = self._load_as_dataframe()
        return list(df.columns)

    def fetch_all(self) -> List[Dict[str, Any]]:
        """Load *all* records into memory (avoid for very large files)."""
        return self._load_as_dataframe().to_dict(orient="records")

    def fetch_stream(self) -> Iterator[Dict[str, Any]]:
        """Row-wise generator (preferred for very large datasets)."""
        yield from self._stream_dicts()

    @staticmethod
    def _Z(conf_level: float) -> float:
        """Z-score lookup for common confidence levels."""
        return {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}[conf_level]

    @staticmethod
    def sample_size_proportion(N: int,
                               conf_level: float = 0.95,
                               margin: float = 0.03,
                               p: float = 0.5) -> int:
        """
        Cochran formula (finite-population correction) for a proportion.

        N : population size
        p : expected proportion (worst-case 0.5 if unknown)
        """
        Z = SampleFileStorage._Z(conf_level)
        n0 = Z ** 2 * p * (1 - p) / margin ** 2
        return math.ceil(n0 / (1 + (n0 - 1) / N))

    @staticmethod
    def sample_size_mean(N: int,
                         conf_level: float = 0.95,
                         margin: float = 1.0,
                         sigma: float = 10.0) -> int:
        """
        Cochran formula for estimating a mean.
        sigma : population standard deviation (use pilot estimate)
        """
        Z = SampleFileStorage._Z(conf_level)
        n0 = Z ** 2 * sigma ** 2 / margin ** 2
        return math.ceil(n0 / (1 + (n0 - 1) / N))

    def rsample(
        self,
        mode: Literal["manual", "proportion", "mean"] = "manual",
        *,
        k: int | None = None,
        conf_level: float = 0.95,
        margin: float = 0.03,
        p: float = 0.5,
        sigma: float = 10.0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Reservoir-sample the current dataset.

        Returns  (sample_records, k)
        """
        N = self.count()

        # Decide sample size -------------------------------------------------
        if mode == "manual":
            if not k or k <= 0:
                raise ValueError("manual mode requires a positive integer k")
        elif mode == "proportion":
            k = self.sample_size_proportion(N, conf_level, margin, p)
        elif mode == "mean":
            k = self.sample_size_mean(N, conf_level, margin, sigma)
        else:
            raise ValueError('mode must be "manual", "proportion", or "mean"')

        self.logger.info(f"Sampling k={k} from N={N} (mode={mode})")

        # Simple reservoir algorithm ----------------------------------------
        reservoir: List[Dict[str, Any]] = []
        for t, rec in enumerate(self.fetch_stream(), start=1):
            if t <= k:
                reservoir.append(rec)
            else:
                j = random.randrange(t)
                if j < k:            # replace with probability k/t
                    reservoir[j] = rec

        return reservoir, k