from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # from .generate.vqa_extract_pdf2img import VQAExtractPdf2Img
    from .generate.vqa_extract_doclayout import VQAExtractDocLayoutMinerU
    # from .generate.vqa_extract_pic_extractor import VQAExtractPicExtractor
    # from .generate.vqa_extract_qapair_extractor import VQAExtractQAPairExtractor
    # from .generate.vqa_extract_tag2img import VQAExtractTag2Img
    # from .generate.vqa_img_helper import VQAClipHeader, VQAConcatenateImages
    from .generate.qa_extract import QAExtractor


else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/vqa/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/vqa/", _import_structure)
