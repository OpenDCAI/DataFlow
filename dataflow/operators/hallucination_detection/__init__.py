"""
Hallucination Detection Operators for DataFlow.

This module provides operators for creating hallucination detection datasets,
including filtering by token length, injecting hallucinations, and parsing
span annotations.

Operators:
- LongContextFilterOperator: Filter samples by token count (8K+, 12K+, etc.)
- HallucinationInjectionOperator: Inject RAGTruth-style hallucinations
- SpanAnnotationOperator: Parse <hal> tags to character positions
- HallucinationDetectionEvaluator: Evaluate hallucination detection models
"""

from dataflow.operators.hallucination_detection.filter.long_context_filter import (
    LongContextFilterOperator,
)
from dataflow.operators.hallucination_detection.generate.hallucination_injection import (
    HallucinationInjectionOperator,
)
from dataflow.operators.hallucination_detection.generate.span_annotation import (
    SpanAnnotationOperator,
)

__all__ = [
    "LongContextFilterOperator",
    "HallucinationInjectionOperator",
    "SpanAnnotationOperator",
]

