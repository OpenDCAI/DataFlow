from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter

    # eval
    from .eval.agenticrag_qaf1_sample_evaluator import AgenticRAGQAF1SampleEvaluator
    # generate
    from .generate.agenticrag_atomic_task_generator import AgenticRAGAtomicTaskGenerator
    from .generate.agenticrag_depth_qa_generator import AgenticRAGDepthQAGenerator
    from .generate.agenticrag_width_qa_generator import AgenticRAGWidthQAGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/agentic_rag/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/agentic_rag/", _import_structure)
