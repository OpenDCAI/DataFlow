from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .generate.rare_doc2query_generator import RAREDoc2QueryGenerator
    from .generate.rare_bm25hardneg_generator import RAREBM25HardNegGenerator
    from .generate.rare_reasondistill_generator import RAREReasonDistillGenerator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/rare/"

    #print(_import_structure)
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/rare/", _import_structure)
