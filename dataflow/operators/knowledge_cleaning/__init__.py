from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .generate.corpus_text_splitter import CorpusTextSplitter
    from .generate.corpus_text_splitter_batch import CorpusTextSplitterBatch
    from .generate.file_or_url_to_markdown_converter import FileOrURLToMarkdownConverter
    from .generate.file_or_url_to_markdown_converter_batch import FileOrURLToMarkdownConverterBatch
    from .generate.knowledge_cleaner import KnowledgeCleaner
    from .generate.knowledge_cleaner_batch import KnowledgeCleanerBatch
    from .generate.multihop_qa_generator import MultiHopQAGenerator
    from .generate.multihop_qa_generator_batch import MultiHopQAGeneratorBatch


else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking



    cur_path = "dataflow/operators/knowledge_cleaning/"


    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/knowledge_cleaning/", _import_structure)
