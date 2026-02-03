import functools
import inspect
import pandas as pd
from dataflow import get_logger

logger = get_logger()


def _normalize_val(v):
    """Normalize List/Dict to string."""
    if isinstance(v, list):
        # List: merge to string.
        # We ensure the result is at least a space " " to avoid being skipped by falsy checks in operators.
        res = "\n\n".join([str(x) for x in v if x])
        return res if res else " "
    if isinstance(v, dict):
        # Dict: try to extract common content fields
        for k in ["content", "text", "raw_content", "cleaned_content", "raw_chunk", "cleaned_chunk"]:
            if k in v:
                res = str(v[k])
                return res if res else " "
        # str({}) is "{}" which is truthy
        return str(v)
    return v


class _CompatStorage:
    """Storage wrapper to auto-normalize specified columns on read."""

    def __init__(self, storage, target_keys):
        self._s = storage
        self._k = target_keys

    def __getattr__(self, name):
        return getattr(self._s, name)

    def read(self, mode):
        data = self._s.read(mode)
        # Only handle DataFrame read
        if mode == 'dataframe' and self._k and isinstance(data, pd.DataFrame):
            for k in self._k:
                if k in data.columns:
                    # Check if conversion is needed (sample check non-null values)
                    # Avoid useless operations on already string columns
                    non_null = data[k].dropna()
                    if not non_null.empty:
                        sample = non_null.iloc[0]
                        if isinstance(sample, (list, dict)):
                            logger.info(
                                f"[_CompatStorage] Auto-normalizing column '{k}' (type: {type(sample).__name__}) to string")
                            data[k] = data[k].apply(_normalize_val)
        return data

    def write(self, d):
        return self._s.write(d)


def _create_run_wrapper(original_run, name):
    @functools.wraps(original_run)
    def new_run(self, storage, *args, **kwargs):
        try:
            # Parse arguments to find all possible input field names
            sig = inspect.signature(original_run)
            bound = sig.bind_partial(self, storage, *args, **kwargs)
            bound.apply_defaults()

            target_keys = []
            for k, v in bound.arguments.items():
                # Heuristic strategy: parameter name contains 'input', 'content', 'key' and value is string, regarded as column name
                if isinstance(v, str) and ('input' in k or 'content' in k or 'key' in k):
                    # Exclude output keys
                    if 'output' not in k:
                        target_keys.append(v)

            # If target columns are found, use wrapped Storage
            if target_keys:
                storage = _CompatStorage(storage, target_keys)
        except Exception as e:
            logger.warning(f"[auto_str_compat] Failed to wrap operator {name}: {e}")

        return original_run(self, storage, *args, **kwargs)

    return new_run


def auto_str_compat(func_or_class):
    """
    Decorator to automatically compatible with List/Dict inputs for operators.
    Can be applied to Operator class or its run method.

    Usage:
        @auto_str_compat
        class MyOperator(OperatorABC):
            ...

    Or:
        class MyOperator(OperatorABC):
            @auto_str_compat
            def run(self, storage, ...):
                ...
    """
    if inspect.isclass(func_or_class):
        # Applied to class: wrap the run method
        if hasattr(func_or_class, 'run'):
            original_run = func_or_class.run
            func_or_class.run = _create_run_wrapper(original_run, func_or_class.__name__)
        return func_or_class
    else:
        # Applied to method
        return _create_run_wrapper(func_or_class, func_or_class.__name__)
