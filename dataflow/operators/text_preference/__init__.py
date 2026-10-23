from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .eval.preference_structure_sample_evaluator import PreferenceStructureSampleEvaluator
    from .filter.preference_structure_filter import PreferenceStructureFilter
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking
    cur_path = "dataflow/operators/text_preference/"
    sys.modules[__name__] = LazyLoader(__name__, cur_path, generate_import_structure_from_type_checking(__file__, cur_path))
