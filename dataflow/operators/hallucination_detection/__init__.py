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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .filter.long_context_filter import LongContextFilterOperator
    from .generate.hallucination_injection import HallucinationInjectionOperator
    from .generate.span_annotation import SpanAnnotationOperator
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/hallucination_detection/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/hallucination_detection/", _import_structure)

