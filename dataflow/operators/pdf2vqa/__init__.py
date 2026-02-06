from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate.mineru_to_llm_input_operator import MinerU2LLMInputOperator
    from .generate.llm_output_parser import LLMOutputParser
    from .generate.qa_merger import QA_Merger
    from .generate.pdf_merger import PDF_Merger


else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/pdf2vqa/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/pdf2vqa/", _import_structure)
