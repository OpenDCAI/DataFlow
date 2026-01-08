"""Generate operators for hallucination detection."""

from dataflow.operators.hallucination_detection.generate.hallucination_injection import (
    HallucinationInjectionOperator,
)
from dataflow.operators.hallucination_detection.generate.span_annotation import (
    SpanAnnotationOperator,
)

__all__ = ["HallucinationInjectionOperator", "SpanAnnotationOperator"]

