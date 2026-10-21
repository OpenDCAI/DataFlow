# Training data preparation (CPU, no model calls)

Three public operators are available from `dataflow.operators.core_text`:

| Operator | Input | Added output |
|---|---|---|
| `TrainingMessagesConverter` | QA, Alpaca, text ShareGPT, or complete messages | A `messages` list, optionally preceded by a system message |
| `TrainingPreferenceConverter` | Mapped text prompt, chosen and rejected | `prompt`, `chosen`, `rejected` columns |
| `TrainingDataValidationFilter` | Single-turn messages, complete multi-turn/tool conversations, or string-valued DPO | `training_data_valid=True` on accepted records |

Defaults retain mapped single-turn QA conversion and single-turn validation.
Select `source_format` for other inputs and `message_profile="conversation"` for
multi-turn/tool validation. Neither conversion nor validation calls a model.
Multimodal content, conversational DPO, and legacy ID-less function-call formats
are outside these profiles; they are not silently flattened or repaired.
Identical chosen/rejected values are not a structural error; quality rules can
be applied separately.

## Direct use

```python
from dataflow.operators.core_text import (
    TrainingMessagesConverter, TrainingPreferenceConverter,
    TrainingDataValidationFilter,
)
from dataflow.utils.storage import FileStorage

storage = FileStorage("input.jsonl", cache_path="./cache/preparation")
convert = TrainingMessagesConverter(system_prompt="Be helpful")
convert.run(storage.step(), input_key="question", input_answer_key="answer")
check = TrainingDataValidationFilter(schema="messages")
check.run(storage.step(), input_key="messages")
print(convert.last_report)  # counts for this call, not the entire pipeline
```

## Alpaca, ShareGPT and multi-turn conversations

Choose the source format explicitly; formats are not auto-detected or mixed in
one invocation. Direct calls use these default columns:

| `source_format` | Required fields | Optional fields and conversion |
|---|---|---|
| `qa` (default) | `prompt`, `answer` | Constant `system_prompt`; QA behavior is unchanged |
| `alpaca` | `instruction`, `output` | `input` joins instruction with `\n\n`; `history` is a list of `[question, answer]` pairs placed first; `system` is placed at the beginning |
| `sharegpt` | `conversations` | Each item has only `from`/`value`; `human` maps to `user`, `gpt` to `assistant`, `system` to `system`. Optional top-level `system` |
| `messages` | `messages` | Validate and copy complete messages, retaining text, tool IDs, argument strings and return values verbatim |

An Alpaca input of `instruction="翻译"`, `input="你好"`, `output="Hello"`
produces `[{"role":"user","content":"翻译\n\n你好"},
{"role":"assistant","content":"Hello"}]`. Historical pairs become earlier
user/assistant turns; they are not merged into the current answer.

```python
storage = FileStorage("alpaca.jsonl", cache_path="./cache/alpaca")
TrainingMessagesConverter(source_format="alpaca").run(
    storage.step(), input_key="instruction", input_answer_key="output",
)
TrainingDataValidationFilter(message_profile="conversation").run(
    storage.step(), input_key="messages",
)
```

For text ShareGPT use `source_format="sharegpt"`, `input_key="conversations"`.
For existing messages use `source_format="messages"`, `input_key="messages"`;
use `output_key="prepared_messages"` if a separate copy is needed. To validate
existing messages without conversion, call the validation operator directly.

For nonstandard columns, map `input_key`, `input_answer_key`, and Alpaca's
`input_context_key`/`input_history_key`; `input_system_key` can map a system
column for any format. Missing, null, or blank optional text fields are omitted;
missing/null history means no history. Populated fields of the wrong type fail
validation. Text that is used is kept verbatim, not stripped. A source system
message and a second system setting conflict; neither silently overwrites the
other. ShareGPT `function_call`/`observation` variants require a separate adapter
because they do not necessarily provide the call IDs this profile requires.

Inside a Pipeline, pass consumed input keys explicitly so dependency checking can
see them. Only include optional key mappings in the compiled graph when those
columns exist in the dataset. Direct calls can use the defaults shown above.

## Conversation and tool-call rules

`message_profile="conversation"` validates a complete training conversation:

- At most one nonblank system message, at the beginning; then user/assistant
  turns. Each user turn must end with a nonblank assistant answer.
- An assistant may issue one or more `tool_calls` before answering. Each call has
  a unique `id`, `type="function"`, and `function` with nonblank `name` and
  `arguments` containing a JSON object encoded as a string.
- Each call receives exactly one `role="tool"` response with the matching
  `tool_call_id`, after the call and before assistant continues. Parallel call
  responses may arrive in any order. More tool rounds within a turn are allowed.
- Assistant content may be missing, null or empty when issuing tool calls. Tool
  response content must be a string and may be empty. Ordinary user/assistant
  content must be nonblank text.
- Missing, duplicate and unexpected responses, reused call IDs, invalid roles,
  incomplete turns and unsupported nested fields are rejected with field paths.

This checks training structure. It does not execute tools, validate arguments
against an external tool schema, check top-level `tools` definitions, assess
answer truthfulness, or prove that training will improve a model. Other top-level
source columns are retained; callers must separately validate any extra metadata
they use. Unknown nested message/call fields are rejected rather than discarded.

The default `message_profile="single_turn"` still accepts only optional system,
user, assistant text messages. Existing strict behavior is unchanged. Both
profiles use the same fail-fast/quarantine contract below.

## DPO

DPO conversion with mapped columns:

```python
convert = TrainingPreferenceConverter()
convert.run(
    storage.step(), input_key="question",
    input_chosen_key="preferred", input_rejected_key="dispreferred",
)
TrainingDataValidationFilter(schema="dpo").run(
    storage.step(), input_prompt_key="prompt",
    input_chosen_key="chosen", input_rejected_key="rejected",
)
```

Use a separate Storage/input for each independent example. Pass active validation
input keys explicitly inside `Pipeline.forward()` so the compiler can check their
dependencies. Converter inputs/outputs already have explicit key parameters.

Source columns and order are retained. Requested output columns are overwritten
if they exist; use different output keys if the originals must be retained.
Text is preserved verbatim, including whitespace; blank-only strings are rejected.
Reasoning is never implicitly merged into the answer. Export only the desired
training columns as a separate final step if required by the trainer.

## Errors and quarantine

The default `on_invalid="raise"` raises a `TrainingDataError` at the first invalid
record and does not write any output for that call. The exception has `field` and
`code` attributes. Configuration, filesystem, and programming errors propagate;
they are not treated as bad samples.

To continue, set `on_invalid="quarantine", errors_path="run/errors.jsonl"` on a
converter or validator. Valid records are written, invalid records are excluded,
and errors append to JSONL. An error includes operator, step (when available),
one-based row position, field path, code and message. Raw inputs are not logged.
Use a separate error file per stage/run; ledgers do not deduplicate retries.
Their location must differ from the current Storage input/output and source.

Row positions refer to **this step's input**, not the original source after an
upstream filter changes its rows. With batched Storage they include the input
batch offset. Retain a source ID column for application-level reconciliation.
This initial API does not promise source replay, transactional error/output
commits, or recovery. `last_report` describes validation in the latest invocation;
it does not establish that a write committed if an exception interrupted it.

When every row is quarantined, an empty DataFrame with the output schema is
written. Ordinary, lazy, and streaming Storage are tested on nonempty examples;
lazy callers must flush before reading output files. No existing Storage defaults
or Pipeline scheduling behavior are changed.

## Offline example

Run the packaged example module using a new output directory:

```bash
python -m dataflow.statics.pipelines.cpu_pipelines.training_data_preparation \
  --output-dir /tmp/dataflow-training-example
```

It creates three source records, writes two valid training records and one error.
For Alpaca history, ShareGPT multi-turn and tool-call examples:

```bash
python -m dataflow.statics.pipelines.cpu_pipelines.conversation_training_preparation \
  --output-dir /tmp/dataflow-conversation-example
```

The second example runs three actual conversion/validation Pipelines. For each
format it writes one valid record and one error, checks complete messages against
fixed expected outputs, verifies error field/code, and checks that source bytes
were not changed. Inspect `summary.json` and each format's `training.jsonl` and
`conversion_errors.jsonl`. Both examples need no GPU, API key or Pyromind service.

Tests:

```bash
python -m pytest test/cpu_only/test_training_data_preparation.py \
  test/cpu_only/test_training_conversations.py -q
```

These are functional assertions, including expected content, rejected records,
error locations and file/lazy/batch/stream Pipeline output. They check more than
whether code runs. They cover the documented profiles and constructed fixtures,
not every data dialect, real model service or downstream training framework.

For opt-in JSONL read quarantine, enhanced batch resume, source-linked model
failures and replay, see [resilient JSONL processing](resilient_jsonl_processing.md).
These are separate APIs; the training converters' own error ledgers still do not
provide a replay/transaction contract.
