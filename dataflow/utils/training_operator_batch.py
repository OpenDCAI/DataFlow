"""Training-data batch helpers for row-preserving transforms and validation."""
import pandas as pd
from dataflow.utils.training_data import check_keys, TrainingDataError


def require_columns(frame, input_columns):
    """Require configured input fields to exist in a DataFrame."""
    check_keys(*input_columns)
    for key in input_columns:
        if key not in frame.columns:
            raise TrainingDataError(key, "missing_field", "Required input column is missing")


def _validate_output_columns(frame, output_columns):
    check_keys(*output_columns)
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("Output fields must be distinct")
    if any(key in frame.columns for key in output_columns):
        raise ValueError("Output fields must not overwrite existing columns")


def transform_rows(storage, transform, output_columns, input_columns=()):
    """Apply a row transform and append new columns after the full batch succeeds."""
    frame = storage.read("dataframe")
    require_columns(frame, input_columns)
    _validate_output_columns(frame, output_columns)
    records = frame.to_dict("records") if len(frame.columns) else [{} for _ in range(len(frame))]
    updates = [transform(row) for row in records]
    result = frame.copy()
    for key in output_columns:
        result[key] = pd.Series([row[key] for row in updates], index=frame.index, dtype=object)
    storage.write(result)
    return output_columns


def evaluate_rows(storage, validator, *, output_valid_key, output_error_key, input_columns=()):
    """Evaluate training-data rows and append aligned validity/error columns.

    The validator must raise ``TrainingDataError`` for invalid content. Every
    input row is retained: valid rows receive ``True`` and ``None``; invalid
    rows receive ``False`` and a structured error. Configuration, missing
    columns, output conflicts and storage errors are raised to the caller.
    """
    frame = storage.read("dataframe")
    require_columns(frame, input_columns)
    output_columns = [output_valid_key, output_error_key]
    _validate_output_columns(frame, output_columns)
    locations = frame.attrs.get("dataflow_source_locations")
    records = frame.to_dict("records") if len(frame.columns) else [{} for _ in range(len(frame))]
    valid_values = []
    error_values = []
    for position, row in enumerate(records):
        try:
            validator(row)
        except TrainingDataError as error:
            detail = {"field": error.field, "code": error.code, "message": str(error),
                      "row": position + 1 + (getattr(storage, "batch_step", 0) * (getattr(storage, "batch_size", None) or 0))}
            if locations is not None:
                detail["source_location"] = locations[position]
            valid_values.append(False)
            error_values.append(detail)
        else:
            valid_values.append(True)
            error_values.append(None)
    result = frame.copy()
    result[output_valid_key] = pd.Series(valid_values, index=frame.index, dtype=object)
    result[output_error_key] = pd.Series(error_values, index=frame.index, dtype=object)
    storage.write(result)
    return output_columns


def filter_validation_rows(storage, input_key):
    """Keep rows whose validation flag is the boolean value ``True``."""
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
