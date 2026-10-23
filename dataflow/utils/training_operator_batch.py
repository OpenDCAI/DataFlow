"""Batch primitives: declared fields, aligned results, explicit filtering."""
import pandas as pd
from dataflow.utils.training_data import check_keys, TrainingDataError


def require_columns(frame, input_columns):
    check_keys(*input_columns)
    for key in input_columns:
        if key not in frame.columns:
            raise TrainingDataError(key, "missing_field", "Required input column is missing")


def transform_rows(storage, transform, output_columns, input_columns=()):
    frame = storage.read("dataframe")
    require_columns(frame, input_columns)
    check_keys(*output_columns)
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("Output fields must be distinct")
    if any(key in frame.columns for key in output_columns):
        raise ValueError("Output fields must not overwrite existing columns")
    records = frame.to_dict("records") if len(frame.columns) else [{} for _ in range(len(frame))]
    updates = [transform(row) for row in records]
    result = frame.copy()
    for key in output_columns:
        result[key] = pd.Series([row[key] for row in updates], index=frame.index, dtype=object)
    storage.write(result)
    return output_columns


def evaluate_rows(storage, validate, output_valid_key, output_error_key, input_columns=()):
    frame = storage.read("dataframe")
    require_columns(frame, input_columns)
    locations = frame.attrs.get("dataflow_source_locations")
    counter = iter(range(len(frame)))
    def evaluate(row):
        position = next(counter)
        try:
            validate(row)
        except TrainingDataError as error:
            detail = {"field": error.field, "code": error.code, "message": str(error),
                      "row": position + 1 + (getattr(storage, "batch_step", 0) * (getattr(storage, "batch_size", None) or 0))}
            if locations is not None:
                detail["source_location"] = locations[position]
            return {output_valid_key: False, output_error_key: detail}
        return {output_valid_key: True, output_error_key: None}
    # One read: lazy/stream storage need not support an extra stateful read.
    class Batch:
        def read(self, kind): return frame
        def write(self, result): return storage.write(result)
    return transform_rows(Batch(), evaluate, [output_valid_key, output_error_key], input_columns)


def filter_validation_rows(storage, input_key):
    frame = storage.read("dataframe")
    require_columns(frame, [input_key])
    if not all(pd.api.types.is_bool(value) for value in frame[input_key]):
        raise ValueError("Validation flags must be boolean, not strings or missing values")
    mask = frame[input_key].astype(bool)
    result = frame.loc[mask].copy()
    locations = frame.attrs.get("dataflow_source_locations")
    if locations is not None:
        result.attrs["dataflow_source_locations"] = [location for location, keep in zip(locations, mask) if keep]
    storage.write(result)
    return []
