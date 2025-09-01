from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from filter.content_chooser import ContentChooser
    # eval
    from eval.f1_scorer import F1Scorer
    # generate
    from generate.atomic_task_generator import AtomicTaskGenerator
    from generate.auto_prompt_generator import AutoPromptGenerator
    from generate.depth_qa_generator import DepthQAGenerator
    from generate.qa_generator import QAGenerator
    from generate.qa_scorer import QAScorer
    from generate.width_qa_generator import WidthQAGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/agentic_rag/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/agentic_rag/", _import_structure)
