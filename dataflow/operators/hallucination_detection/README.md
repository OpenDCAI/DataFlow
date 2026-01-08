# Hallucination Detection Operators

This module provides DataFlow operators for creating hallucination detection datasets.

## Operators

### LongContextFilterOperator

Filters samples by token count for long-context evaluation datasets.

```python
from dataflow.operators.hallucination_detection import LongContextFilterOperator

filter_op = LongContextFilterOperator(
    tokenizer_name="answerdotai/ModernBERT-base",
    min_tokens=8000,
    max_tokens=24000,
)
```

**Parameters:**
- `tokenizer`: Pre-loaded HuggingFace tokenizer (optional)
- `tokenizer_name`: Model name to load tokenizer from (default: "answerdotai/ModernBERT-base")
- `min_tokens`: Minimum token count (default: 8000)
- `max_tokens`: Maximum token count (default: 32000)
- `text_fields`: List of fields to count tokens for (default: ["prompt", "answer"])

### HallucinationInjectionOperator

Injects RAGTruth-style hallucinations into LLM-generated answers.

```python
from dataflow.operators.hallucination_detection import HallucinationInjectionOperator
from dataflow.serving import LocalHostLLMAPIServing_vllm

llm = LocalHostLLMAPIServing_vllm(
    hf_model_name_or_path="Qwen/Qwen2.5-72B-Instruct",
    vllm_server_port=8000,
)

inject_op = HallucinationInjectionOperator(
    llm_serving=llm,
    hallucination_ratio=0.5,
    hallucination_types=["Evident Conflict", "Evident Baseless Info"],
)
```

**Hallucination Types:**
- `Evident Conflict`: Direct contradiction of source facts
- `Evident Baseless Info`: Fabricated facts not in source
- `Subtle Baseless Info`: Implied but unstated claims
- `Subtle Conflict`: Nuanced contradictions

**Parameters:**
- `llm_serving`: LLM serving backend for generation
- `hallucination_ratio`: Fraction of samples to inject hallucinations (0-1)
- `hallucination_types`: List of hallucination types to use
- `seed`: Random seed for reproducibility

### SpanAnnotationOperator

Converts document-level labels to span-level using NLI.

```python
from dataflow.operators.hallucination_detection import SpanAnnotationOperator

annotator = SpanAnnotationOperator(
    nli_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    contradiction_threshold=0.7,
)
```

**Parameters:**
- `nli_model`: HuggingFace model for NLI
- `contradiction_threshold`: Threshold for labeling as contradiction
- `device`: Device to run on ("cuda" or "cpu")

## Example Pipeline

See `dataflow/example/HallucinationDetectionPipeline/example_pipeline.py` for a complete example.

```python
from dataflow.pipeline import PipelineABC
from dataflow.operators.hallucination_detection import (
    LongContextFilterOperator,
    HallucinationInjectionOperator,
)

class HallucinationPipeline(PipelineABC):
    def __init__(self, llm_serving):
        super().__init__()
        self.filter = LongContextFilterOperator(min_tokens=8000)
        self.inject = HallucinationInjectionOperator(llm_serving=llm_serving)
    
    def forward(self):
        self.filter.run(storage=self.storage, input_key="data", output_key="filtered")
        self.inject.run(storage=self.storage, input_key="filtered", output_key="output")
```

## Output Format

The operators produce datasets compatible with hallucination detection training:

```json
{
    "prompt": "Context: ... Question: ...",
    "answer": "The answer text with hallucination.",
    "has_hallucination": true,
    "hallucination_type": "Evident Conflict",
    "labels": [
        {
            "text": "hallucinated span",
            "start": 10,
            "end": 28,
            "label": "hallucinated"
        }
    ],
    "num_tokens": 15234
}
```

## Related Resources

- [32K ModernBERT Hallucination Detector](https://huggingface.co/llm-semantic-router/modernbert-base-32k-haldetect)
- [Long-Context Evaluation Dataset](https://huggingface.co/datasets/llm-semantic-router/longcontext-haldetect)
- [RAGTruth Paper](https://aclanthology.org/2024.acl-long.585/)

