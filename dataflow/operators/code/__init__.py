from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.autogen_filter import AutogenFilter
    from .filter.code_length_filter import CodeLengthFilter
    from .filter.code_pattern_filter import DataPatternFilter
    from .filter.code_quality_filter import CodeTextQualityFilter
    from .filter.doc_quality_filter import DocQualityFilter
    from .filter.file_type_filter import FileTypeFilter
    from .filter.score_filter import ScoreFilter
    
    # generate
    from .generate.code_generator import CodeGenerator
    from .generate.instruction_synthesizer import InstructionSynthesizer
    
    # eval
    from .eval.pair_scorer import PairScorer
    from .eval.sandbox_validator import SandboxValidator
    from .eval.shared_vis_python_exe import (
        ImageRuntime,
        DateRuntime,
        CustomDict,
        ColorObjectRuntime,
        PythonExecutor
    )

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/code/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/code/", _import_structure)
