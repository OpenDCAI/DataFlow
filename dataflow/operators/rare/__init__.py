from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .generate.doc_to_query import Doc2Query
    from .generate.bm25_hard_negative import BM25HardNeg
    from .generate.reason_distill import ReasonDistill

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/rare/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/rare/", _import_structure)
