from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # GeneralText
    from .GeneralText.pretrain_generator import PretrainGenerator
    from .GeneralText.sft_generator import SupervisedFinetuneGenerator
    from .GeneralText.prompted_generator import PromptedGenerator

    # Reasoning
    from .Reasoning.answer_generator import AnswerGenerator
    from .Reasoning.question_category_classifier import QuestionCategoryClassifier
    from .Reasoning.question_difficulty_classifier import QuestionDifficultyClassifier
    from .Reasoning.question_generator import QuestionGenerator
    from .Reasoning.answer_extraction_qwenmatheval import AnswerExtraction_QwenMathEval
    from .Reasoning.pseudo_answer_generator import PseudoAnswerGenerator
    from .Reasoning.pretrain_format_converter import PretrainFormatConverter

    # Text2SQL
    from .Text2SQL.DatabaseSchemaExtractor import DatabaseSchemaExtractor
    from .Text2SQL.ExtraKnowledgeGenerator import ExtraKnowledgeGenerator
    from .Text2SQL.PromptGenerator import PromptGenerator
    from .Text2SQL.QuestionRefiner import QuestionRefiner
    from .Text2SQL.SchemaLinking import SchemaLinking
    from .Text2SQL.SQLDifficultyClassifier import SQLDifficultyClassifier
    from .Text2SQL.SQLFilter import SQLFilter
    from .Text2SQL.Text2SQLDifficultyClassifier import Text2SQLDifficultyClassifier

    # KnowledgeCleaning
    from .KnowledgeCleaning.corpus_text_splitter import CorpusTextSplitter
    from .KnowledgeCleaning.knowledge_extractor import KnowledgeExtractor
    from .KnowledgeCleaning.pdf_extractor import PDFExtractor
    from .KnowledgeCleaning.knowledge_cleaner import KnowledgeCleaner
    from .KnowledgeCleaning.multihop_qa_generator import MultiHopQAGenerator

    # AgenticRAG
    from .AgenticRAG.auto_prompt_generator import AutoPromptGenerator
    from .AgenticRAG.qa_scorer import QAScorer
    from .AgenticRAG.qa_generator import QAGenerator
    from .AgenticRAG.atomic_task_generator import AtomicTaskGenerator
    from .AgenticRAG.depth_qa_generator import DepthQAGenerator
    from .AgenticRAG.width_qa_generator import WidthQAGenerator

    # RARE
    from .RARE.doc_to_query import Doc2Query
    from .RARE.bm25_hard_negative import BM25HardNeg
    from .RARE.reason_distill import ReasonDistill
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    # from .Reasoning import *
    # from .GeneralText import *
    # from .Text2SQL import *

    # from .KnowledgeCleaning import *
    # from .AgenticRAG import *
    # from .RARE import *


    cur_path = "dataflow/operators/generate/"

    # _import_structure = {
    #     # GeneralText
    #     "PretrainGenerator": (cur_path + "GeneralText/pretrain_generator.py", "PretrainGenerator"),
    #     "SupervisedFinetuneGenerator": (cur_path + "GeneralText/sft_generator.py", "SupervisedFinetuneGenerator"),
    #     "PromptGenerator": (cur_path + "GeneralText/prompt_generator.py", "PromptGenerator"),
        
    #     # Reasoning
    #     "AnswerGenerator": (cur_path + "Reasoning/answer_generator.py", "AnswerGenerator"),
    #     "QuestionCategoryClassifier": (cur_path + "Reasoning/question_category_classifier.py", "QuestionCategoryClassifier"),
    #     "QuestionDifficultyClassifier": (cur_path + "Reasoning/question_difficulty_classifier.py", "QuestionDifficultyClassifier"),
    #     "QuestionGenerator": (cur_path + "Reasoning/question_generator.py", "QuestionGenerator"),
    #     "AnswerExtraction_QwenMathEval": (cur_path + "Reasoning/answer_extraction_qwenmatheval.py", "AnswerExtraction_QwenMathEval"),
    #     "PseudoAnswerGenerator": (cur_path + "Reasoning/pseudo_answer_generator.py", "PseudoAnswerGenerator"),
    #     "PretrainFormatConverter": (cur_path + "Reasoning/pretrain_format_converter.py", "PretrainFormatConverter"),
        
    #     # Text2SQL
    #     "DatabaseSchemaExtractor": (cur_path + "Text2SQL/DatabaseSchemaExtractor.py", "DatabaseSchemaExtractor"),
    #     "ExtraKnowledgeGenerator": (cur_path + "Text2SQL/ExtraKnowledgeGenerator.py", "ExtraKnowledgeGenerator"),
    #     "PromptGenerator": (cur_path + "Text2SQL/PromptGenerator.py", "PromptGenerator"),
    #     "QuestionRefiner": (cur_path + "Text2SQL/QuestionRefiner.py", "QuestionRefiner"),
    #     "SchemaLinking": (cur_path + "Text2SQL/SchemaLinking.py", "SchemaLinking"),
    #     "SQLDifficultyClassifier": (cur_path + "Text2SQL/SQLDifficultyClassifier.py", "SQLDifficultyClassifier"),
    #     "SQLFilter": (cur_path + "Text2SQL/SQLFilter.py", "SQLFilter"),
    #     "Text2SQLDifficultyClassifier": (cur_path + "Text2SQL/Text2SQLDifficultyClassifier.py", "Text2SQLDifficultyClassifier"),
        
    #     # KnowledgeCleaning
    #     "CorpusTextSplitter": (cur_path + "KnowledgeCleaning/corpus_text_splitter.py", "CorpusTextSplitter"),
    #     "KnowledgeExtractor": (cur_path + "KnowledgeCleaning/knowledge_extractor.py", "KnowledgeExtractor"),
    #     "KnowledgeCleaner": (cur_path + "KnowledgeCleaning/knowledge_cleaner.py", "KnowledgeCleaner"),
    #     "MultiHopQAGenerator": (cur_path + "KnowledgeCleaning/multihop_qa_generator.py", "MultiHopQAGenerator"),

    #     # AgenticRAG
    #     "AutoPromptGenerator": (cur_path + "AgenticRAG/auto_prompt_generator.py", "AutoPromptGenerator"),
    #     "QAScorer": (cur_path + "AgenticRAG/qa_scorer.py", "QAScorer"),
    #     "QAGenerator": (cur_path + "AgenticRAG/qa_generator.py", "QAGenerator"),
    #     "AtomicTaskGenerator": (cur_path + "AgenticRAG/atomic_task_generator.py", "AtomicTaskGenerator"),
    #     "DepthQAGenerator": (cur_path + "AgenticRAG/depth_qa_generator.py", "DepthQAGenerator"),
    #     "WidthQAGenerator": (cur_path + "AgenticRAG/width_qa_generator.py", "WidthQAGenerator"),
        
    #     # RARE
    #     "Doc2Query": (cur_path + "RARE/doc_to_query.py", "Doc2Query"),
    #     "BM25HardNeg": (cur_path + "RARE/bm25_hard_negative.py", "BM25HardNeg"),
    #     "ReasonDistill": (cur_path + "RARE/reason_distill.py", "ReasonDistill"),
    # }
    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)
