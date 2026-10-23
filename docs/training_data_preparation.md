# Training data preparation

These deterministic operators do not call a model. Source validation, filtering,
and conversion are explicit steps, so conversion never silently removes rows.
The unpublished core_text interfaces from the initial PR have been removed.
Existing upstream operators are unchanged.

## Public interfaces and classification

| Domain | Operator | Responsibility |
|---|---|---|
| text_sft/eval | QAStructureSampleEvaluator | Check explicit prompt/answer fields |
| text_sft/eval | AlpacaStructureSampleEvaluator | Check instruction/output and explicitly enabled optional fields |
| text_sft/generate | QAMessagesGenerator | Add a messages column from QA |
| text_sft/generate | AlpacaMessagesGenerator | Add a messages column from Alpaca |
| conversations/eval | ShareGPTStructureSampleEvaluator | Check text from/value ShareGPT |
| conversations/generate | ShareGPTMessagesGenerator | Add messages from text ShareGPT |
| conversations/eval | ConversationStructureSampleEvaluator | Check complete messages/tool relationships |
| conversations/filter | ConversationStructureFilter | Keep True validation flags, leaving retained rows unchanged |
| text_preference/eval | PreferenceStructureSampleEvaluator | Check string-valued prompt/chosen/rejected |
| text_preference/filter | PreferenceStructureFilter | Keep explicitly validated preference records |

The structured field builders use the domain `generate` category because they
add a field without changing row count. The proposed `text_preference` domain
separates preference data from SFT. These are explicit classification proposals
for maintainer review, not claims of prior maintainer approval.

## QA pipeline

```python
from dataflow.pipeline import PipelineABC
from dataflow.utils.storage import FileStorage
from dataflow.operators.text_sft import QAStructureSampleEvaluator, QAMessagesGenerator
from dataflow.operators.conversations import ConversationStructureFilter

class Prepare(PipelineABC):
    def __init__(self, source, cache):
        super().__init__()
        self.storage = FileStorage(source, cache_path=cache)
        self.check = QAStructureSampleEvaluator()
        self.keep = ConversationStructureFilter()
        self.convert = QAMessagesGenerator()

    def forward(self):
        self.check.run(self.storage.step(), input_key="question", input_answer_key="answer")
        self.keep.run(self.storage.step(), input_key="source_valid")
        self.convert.run(self.storage.step(), input_key="question", input_answer_key="answer")

# pipeline = Prepare("source.jsonl", "cache")
# pipeline.compile()  # missing input columns fail before processing
# pipeline.forward()
```

For 100 inputs with three invalid records, evaluation emits 100 rows with
`source_valid` and `source_error`, filtering emits 97, and conversion emits 97.
The evaluation cache retains rejected records for inspection. Errors include
field path, stable code, message and position in the current input; a reader
policy can additionally supply physical source locations. Error messages do
not include raw question/answer contents. Evaluation output contains source
records and should be handled like the input dataset.

Conversion preserves original columns and content, including whitespace and
Unicode. It requires a new output column, raises on invalid content, and writes
only after the complete batch succeeds. Evaluation similarly requires new
output columns. Missing input columns, invalid configuration and storage errors
raise rather than being disguised as per-record content failures. Filters only
accept actual boolean flags, not strings, numbers or missing values.

## Alpaca and optional dependencies

```python
from dataflow.operators.text_sft import AlpacaMessagesGenerator

converter = AlpacaMessagesGenerator()
# Required defaults: instruction/output. Optional defaults are disabled.
converter.run(storage, input_context_key="input", input_history_key="history",
              input_system_key="system", output_key="messages")
```

Use the same optional keys and constructor system_prompt in
`AlpacaStructureSampleEvaluator` before filtering. `None` means the column is not
read. If enabled, the column must exist; null optional values within a record
remain supported. There is no automatic search for history/input/system.
Instruction and nonblank context are joined by two newlines, with historical
turns first. Conflicting source and constructor system prompts are rejected.

## ShareGPT and complete conversations

```python
from dataflow.operators.conversations import (
    ShareGPTStructureSampleEvaluator, ShareGPTMessagesGenerator,
    ConversationStructureSampleEvaluator,
)
```

ShareGPT defaults to `input_key="conversations"`, `output_key="messages"`.
`input_system_key=None` disables a top-level system column. Only text from/value
messages are supported. Unknown nested fields, including training weights, are
rejected. An explicit dataset adapter may add a from/value view while retaining
original metadata; this does not implement loss-weight semantics.

For existing messages no identity-conversion operator is needed. Evaluate them
directly with `ConversationStructureSampleEvaluator(profile="conversation")`,
then filter if desired. `profile="single_turn"` is an explicit stricter option.
Roles, turn order, tool-call IDs and JSON arguments, missing/duplicate returns,
and complete final answers are checked. Parallel tool returns may arrive out
of order; no tool IDs or missing responses are invented.

## Preference data

```python
from dataflow.utils.training_data_formats import map_preference_fields
from dataflow.operators.text_preference import (
    PreferenceStructureSampleEvaluator, PreferenceStructureFilter,
)

mapped = map_preference_fields(
    {"question": "Q", "good": "A", "bad": "B", "id": 1},
    input_prompt_key="question", input_chosen_key="good", input_rejected_key="bad",
)
# mapped retains the source fields and adds prompt/chosen/rejected.
```

Field mapping is a Python utility, not an operator that implicitly filters.
It preserves both answers verbatim and rejects overwriting unrelated source
columns. Evaluation can directly use custom input_prompt_key/input_chosen_key/
input_rejected_key names; no prior field mapping is required for compilation.
System/context concatenation, chat templates and conversational/multimodal DPO
are not performed automatically.

## Migration from the first PR revision

| Removed unpublished interface | Replacement |
|---|---|
| TrainingMessagesConverter | QA/Alpaca/ShareGPT domain builders with fixed field signatures |
| messages identity-conversion mode | Direct conversation evaluation; retain the original messages |
| TrainingDataValidationFilter | Domain evaluator followed by an explicit flag filter |
| TrainingPreferenceConverter | map_preference_fields utility plus preference evaluator/filter |
| on_invalid="quarantine" inside conversion | Source evaluation → explicit filtering → strict conversion |

The old core_text names are not registered or re-exported. They had not been
part of an upstream release, so no misleading core compatibility aliases are
kept. Each new operator's get_desc("zh"/"en") documents actual defaults, optional
fields, row behavior and errors for users and agents.

Run the CPU examples without credentials:

```bash
python -m dataflow.statics.pipelines.cpu_pipelines.training_data_preparation --output-dir /tmp/qa-example
python -m dataflow.statics.pipelines.cpu_pipelines.conversation_training_preparation --output-dir /tmp/conversation-example
```

Use fresh output directories. File, lazy, batch and stream integrations are
covered by regression tests, including wholly invalid first batches and empty
outputs. See [resilient JSONL processing](resilient_jsonl_processing.md) for
reader quarantine, checkpoint and independent request replay boundaries.
