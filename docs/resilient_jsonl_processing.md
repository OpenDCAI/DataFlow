# Resilient local JSONL processing

These are opt-in APIs. Default Storage reads, legacy batch resume, and Serving
list return values keep their existing behavior. This implementation does not
include global deduplication or distributed execution.

## Isolate malformed input records

```python
from dataflow.utils.jsonl_reader import JSONLReadPolicy
from dataflow.utils.storage import StreamBatchedFileStorage

policy = JSONLReadPolicy("run/read-errors.jsonl")
storage = StreamBatchedFileStorage(
    "input.jsonl", cache_path="run/cache", jsonl_read_policy=policy,
)
```

`FileStorage`, `LazyFileStorage`, `BatchedFileStorage`, and
`StreamBatchedFileStorage` accept the same optional policy. It applies only to
local initial JSONL inputs, not remote datasets or intermediate caches. Omit it
for the existing pandas reader and default error behavior.

The new reader rejects blank lines, invalid UTF-8/JSON, nonstandard NaN/Infinity
constants, and JSON values that are not objects. Filesystem errors still abort.
It publishes an atomic error JSONL with physical line numbers and byte offsets,
without raw data. Header inspection, counting, and repeated reads replace this
ledger rather than append duplicates. Use one policy and error path per input.

`policy.last_report` has scanned/accepted/rejected counts. A constant-memory
inspection pass gathers column names and errors before the data pass. Streaming
chunks contain at most the requested number of accepted records; the original
source locations are available in `frame.attrs["dataflow_source_locations"]`.
Columns are not added to the user's data. The training converters include this
location in errors when available. Attributes do not survive JSONL writes:
subsequent stages refer to their own materialized input and should retain source
IDs if the application needs ancestry across stages.

The explicit reader uses Python JSON values and pandas DataFrame construction;
its type/date inference may differ from `pd.read_json`. Input is expected to stay
unchanged during a run. File stat checks invalidate inspection caches; a new run
or enhanced checkpoint initialization uses full input hashes.
