from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .generate.kbc_chunk_generator import KBCChunkGenerator
    from .generate.kbc_chunk_generator_batch import KBCChunkGeneratorBatch
    from .generate.file_or_url_to_markdown_converter import FileOrURLToMarkdownConverter
    from .generate.file_or_url_to_markdown_converter_batch import FileOrURLToMarkdownConverterBatch
    from .generate.kbc_text_cleaner import KBCTextCleaner
    from .generate.kbc_text_cleaner_batch import KBCTextCleanerBatch
    # from .generate.mathbook_question_extract import MathBookQuestionExtract
    # from .generate.kbc_multihop_qa_generator import KBCMultiHopQAGenerator
    from .generate.kbc_multihop_qa_generator_batch import KBCMultiHopQAGeneratorBatch
    from .generate.math_vqa_extract_pdf2img import MathVQAExtractPdf2Img
    from .generate.math_vqa_extract_doclayout import MathVQAExtractDocLayout, VQAExtractDocLayoutMinerU
    from .generate.math_vqa_extract_pic_extractor import MathVQAExtractPicExtractor
    from .generate.math_vqa_extract_qapair_extractor import MathVQAExtractQAPairExtractor
    from .generate.math_vqa_extract_tag2img import MathVQAExtractTag2Img
    from .generate.math_vqa_img_helper import MathVQAClipHeader, MathVQAConcatenateImages


else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/knowledge_cleaning/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/knowledge_cleaning/", _import_structure)
