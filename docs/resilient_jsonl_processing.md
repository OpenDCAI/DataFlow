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

## Enhanced batch resume

On a compiled `BatchedPipelineABC` or `StreamBatchedPipelineABC`:

```python
pipeline.forward(
    batch_size=100,
    checkpoint_dir="run/checkpoint",
    checkpoint_config={
        "pipeline_revision": 1,
        "prompt": "the complete prompt",
        "model": "model identifier",
        "parser_revision": 1,
    },
)
```

Use fresh output/cache directories for the first run. Restart the same pipeline
with the same arguments to resume. This mode does not read or migrate the legacy
`*_last_success_step.txt`; `resume_step` cannot be combined with it. Without
`checkpoint_dir`, legacy behavior is unchanged. `checkpoint_config` is required:
the caller must declare constructor settings, prompt, model, parser versions and
other dependencies that affect results. Arbitrary Python objects are not inferred
or fingerprinted automatically. The graph's operator types, run arguments,
output paths, batch size, reading mode, and input content hash are also bound.

Supported: a single local JSONL source, standard batched or streaming-batched
JSONL Storage, one writer, and operators that write each output via that Storage.
Custom Storage, lazy buffers, remote outputs, extra operator files, external
side effects and arbitrary stateful operators are not covered. Unix locking and
durability are tested on macOS; Windows code paths have not been validated.

After each batch the output files are flushed, then a versioned manifest is
atomically published with committed byte sizes and progress. On resume, all
committed files are checked before tails are truncated. Uncommitted first-batch
files are removed, including the zero-offset case. Changed input/configuration,
invalid state, missing/short outputs, and a cache with preexisting unowned outputs
cause an error rather than an automatic restart. Empty/filter-all batches are
supported. Full content hashes detect input changes between runs; stat checks
are used during a run. Same-length external corruption of committed output is
not checked with output checksums.

This is output/checkpoint consistency, not exactly-once model billing. Retrying
an uncommitted batch may repeat calls or other external work. Do not place
untracked side effects inside an operator and assume they roll back.

## Per-record failures and replay

`RecordedPromptedGenerator` is a separate opt-in operator; existing generators
are unchanged. It processes each input independently and requires a materialized
local JSONL input (ordinary/batched/streaming Storage, not unflushed lazy input).

```python
from dataflow.operators.core_text import RecordedPromptedGenerator
from dataflow.serving import APILLMServing_request

serving = APILLMServing_request(
    api_url="http://localhost:8000/v1/chat/completions",
    model_name="my-model", record_results=True,
)
generator = RecordedPromptedGenerator(
    serving, ledger_dir="run/generation-ledger",
    run_config={"parser_revision": 1},
    system_prompt="Answer the question",
)
generator.run(storage.step(), input_key="question", output_key="answer")
```

Pass all custom parser/dependency versions in `run_config`. The operator also
binds the system prompt, model/API URL, model options, field mapping, retry count
and request budget. Calls on one Serving instance must not be concurrent across
independent clients: `last_results` belongs to the most recent generation call.
Within a generation call it follows input order, even if requests complete out
of order. It contains error codes, actual attempt counts and elapsed time, not
raw payloads or credentials. In this recorded mode blank text is a failed result.

The ledger stores outcomes in SQLite and publishes `failures.jsonl` and
`report.json`. Failures carry source hash, physical line and resource base.
Counts distinguish success/failed/skipped/pending from request attempts. Generated
updates are saved for successful records, so an output retry can reuse them;
source inputs are referenced, not copied into error logs. A source file and its
relative assets must remain available and unchanged. Generated data can of course
contain sensitive content and should be protected as output data.

Optional `parse_response` may convert a response to a JSON-serializable value.
A `ValueError` means a data/parse failure; programming exceptions propagate. Raw
parser exception messages are not put in failure artifacts. This version executes
one logical generation per record; it does not generalize arbitrary multi-call
agent trajectories. The dedicated generator currently sends records sequentially;
it prioritizes source association and recovery over model throughput.

Replay and merge:

```python
generator.replay_failed(
    "input.jsonl", "run/merged.jsonl",
    input_key="question", output_key="answer",
)
```

Use exactly the stage input and field mapping used originally. For a quarantined
source, supply its `read_policy` too. Replay calls only failed records. Successful
results are reused; repeated replay with no remaining failures makes no model
calls. Output is atomically rebuilt in source order rather than appended, making
repeated merges idempotent. Changed input/configuration is refused. Relative
resources stay based at the original source directory; no subset is moved.

The low-level `RecordLedger` can also run a callback:

```python
from dataflow.utils.record_ledger import RecordLedger, RecordProcessingError

with RecordLedger("run/ledger", "input.jsonl", {"processor_revision": 1}) as ledger:
    ledger.process(processor)                    # missing outcomes only
    ledger.process(processor, only_failed=True)  # recorded failures only
    ledger.export_results("run/merged.jsonl")
```

`processor(row)` returns a dict of generated updates. Expected sample failures
raise `RecordProcessingError(code, stage, attempts)`. For exact successful request
attempt accounting a callback can return `RecordResult(updates, attempts)`.
Other exceptions abort without being hidden. A new process continues missing
records; an in-flight request interrupted before its outcome commits can repeat.
Do not run multiple writers against the same ledger. The checkpoint guarantees
cover pipeline outputs; ledger exports/reports can be rebuilt from SQLite if a
crash occurs between publication steps. The entire pipeline plus ledger is not
a multi-file atomic transaction, and arbitrary side effects are not rolled back.

## Optional request budget

```python
serving = APILLMServing_request(
    model_name="my-model", request_budget_seconds=30, max_workers=4,
    connect_timeout=10, read_timeout=20,
)
```

The budget starts when a thread-pool task begins, **excluding time queued for a
worker slot**. It covers local HTTP worker startup, connection/read, retries and
backoff. A dedicated subprocess makes HTTP requests; at the deadline it is killed
and reaped before the task returns. A heartbeat cannot indefinitely reset the
overall budget. Concurrency is bounded by `max_workers`. Spawn/cleanup scheduling
adds small overhead; this is not a real-time scheduling guarantee. The remote
server's computation/billing cannot be canceled by killing the local worker.

This mode automatically enables `record_results`. It still returns the normal
ordered list with `None` for failed records. The existing `max_retries` convention
is preserved: it is the maximum attempt count. Connection errors can be retried
in budget mode, and unsuccessful request outcomes return `None`. Keep the mode
disabled to retain legacy request exception/retry behavior. A subprocess is created
per logical request, so this mode costs more than reusing in-process HTTP sessions;
choose worker count and budget accordingly. No API secrets appear in command-line
arguments. There are no GPU, Ray, or new third-party dependency requirements.

An HTTP 200 is not sufficient on its own: empty or malformed completions are
retried within the same attempt count and remaining total budget. HTTP failures,
response failures and backoff share that one budget. Reasoning-only responses
retain the legacy think/answer formatting, and embedding responses use the same
formatter in both modes.

## Offline example and tests

```bash
python -m dataflow.statics.pipelines.cpu_pipelines.resilient_jsonl_processing \
  --output-dir /tmp/dataflow-resilient-example
python -m pytest test/cpu_only/test_jsonl_read_policy.py \
  test/cpu_only/test_pipeline_enhanced_checkpoint.py \
  test/cpu_only/test_record_ledger.py \
  test/cpu_only/test_api_recorded_requests.py -q
```

The example makes no network requests: one malformed line is isolated, two of ten
records fail in a fake Serving, only those two are replayed, and ten records are
exported. API budget tests use a temporary localhost HTTP server and need permission
to bind a local port, but no external service or credentials.
