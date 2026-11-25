# Chinese visual CoT generation

`ChineseVisualCoTGenerator` converts a local image, a Chinese question, and a reference answer into a candidate reasoning trace. It filters low-quality and duplicate inputs, verifies candidates, and ranks eligible traces using the mean log probability of their answer tokens. The verifier reviews text against the question and reference answer.

## Input and model services

Provide one JSON object per line. Image paths can be relative to the image root.

```json
{"id":"example-1","images":["example.png"],"question":"图中红色方块有几个？","answer":"3"}
```

Configure Qwen3.5-4B, 9B, 27B, and 122B-A10B endpoints. The 4B model tests difficulty, the last three serve as teachers, the 27B model also verifies responses, and the 122B-A10B model cleans questions and answers. Teacher endpoints must support `prompt_logprobs`, `return_token_ids`, and continuation of the assistant message. Set `VISUAL_COT_API_KEY` in the runtime environment.

```python
import os

from dataflow.operators.core_vision import ChineseVisualCoTGenerator
from dataflow.utils.storage import FileStorage
from dataflow.utils.visual_cot.preprocessing import BestEffortImageFilter
from dataflow.utils.visual_cot.serving import LocalCLIPEncoder, VisualCoTAPIServing, VisualCoTServingBundle


def serving(size, purpose="generation"):
    model_id = f"Qwen/Qwen3.5-{size}"
    endpoint = os.environ["VISUAL_COT_" + size.replace("-", "_") + "_URL"]
    return VisualCoTAPIServing(model_id, model_id, endpoint, purpose=purpose)


encoder = LocalCLIPEncoder("/path/to/clip-weights")
models = VisualCoTServingBundle(
    small_model=serving("4B"),
    teacher_servings=[serving(size) for size in ("9B", "27B", "122B-A10B")],
    verifier_serving=serving("27B", "verifier"),
    cleaner_serving=serving("122B-A10B", "cleaner"),
    auxiliary_resources=[encoder],
)
storage = FileStorage("qa.jsonl", cache_path="visual_cot_cache")
operator = ChineseVisualCoTGenerator(
    models, BestEffortImageFilter(encoder, use_faiss=False), image_root="./images"
)
try:
    output_key = operator.run(storage.step())
finally:
    models.cleanup()
```

The operator preserves every input row and returns the name of an added column, `visual_cot_result` by default. Its status is `accepted`, `rejected`, or `invalid_input`. Accepted records contain the Chinese trace, final answer, and training messages. Set `journal_dir` to reuse results after interruptions under the same input and configuration.
