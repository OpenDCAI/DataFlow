import sys
from dataflow.utils.registry import LazyLoader

# from .Reasoning import *
# from .GeneralText import *
# from .Text2SQL import *

# from .KnowledgeCleaning import *
# from .AgenticRAG import *
# from .RARE import *


cur_path = "dataflow/operators/generate/"
_import_structure = {
    # GeneralText
    "PretrainGenerator": (cur_path + "GeneralText/pretrain_generator.py", "PretrainGenerator"),
    "SupervisedFinetuneGenerator": (cur_path + "GeneralText/sft_generator.py", "SupervisedFinetuneGenerator"),
    "PromptedGenerator": (cur_path + "GeneralText/prompted_generator.py", "PromptedGenerator"),

    # Reasoning
    "AnswerGenerator": (cur_path + "Reasoning/answer_generator.py", "AnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "Reasoning/question_category_classifier.py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "Reasoning/question_difficulty_classifier.py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "Reasoning/question_generator.py", "QuestionGenerator"),
    "AnswerExtraction_QwenMathEval": (cur_path + "Reasoning/answer_extraction_qwenmatheval.py", "AnswerExtraction_QwenMathEval"),
    "PseudoAnswerGenerator": (cur_path + "Reasoning/pseudo_answer_generator.py", "PseudoAnswerGenerator"),
    "PretrainFormatConverter": (cur_path + "Reasoning/pretrain_format_converter.py", "PretrainFormatConverter"),
    
    # Text2SQL
    "DatabaseSchemaExtractor": (cur_path + "Text2SQL/DatabaseSchemaExtractor.py", "DatabaseSchemaExtractor"),
    "ExtraKnowledgeGenerator": (cur_path + "Text2SQL/ExtraKnowledgeGenerator.py", "ExtraKnowledgeGenerator"),
    "PromptGenerator": (cur_path + "Text2SQL/PromptGenerator.py", "PromptGenerator"),
    "QuestionRefiner": (cur_path + "Text2SQL/QuestionRefiner.py", "QuestionRefiner"),
    "SchemaLinking": (cur_path + "Text2SQL/SchemaLinking.py", "SchemaLinking"),
    "SQLDifficultyClassifier": (cur_path + "Text2SQL/SQLDifficultyClassifier.py", "SQLDifficultyClassifier"),
    "SQLFilter": (cur_path + "Text2SQL/SQLFilter.py", "SQLFilter"),
    "Text2SQLDifficultyClassifier": (cur_path + "Text2SQL/Text2SQLDifficultyClassifier.py", "Text2SQLDifficultyClassifier"),
    
    # KnowledgeCleaning
    "CorpusTextSplitter": (cur_path + "KnowledgeCleaning/corpus_text_splitter.py", "CorpusTextSplitter"),
    "KnowledgeExtractor": (cur_path + "KnowledgeCleaning/knowledge_extractor.py", "KnowledgeExtractor"),
    "KnowledgeCleaner": (cur_path + "KnowledgeCleaning/knowledge_cleaner.py", "KnowledgeCleaner"),
    "MultiHopQAGenerator": (cur_path + "KnowledgeCleaning/multihop_qa_generator.py", "MultiHopQAGenerator"),

    # AgenticRAG
    "AutoPromptGenerator": (cur_path + "AgenticRAG/auto_prompt_generator.py", "AutoPromptGenerator"),
    "QAScorer": (cur_path + "AgenticRAG/qa_scorer.py", "QAScorer"),
    "QAGenerator": (cur_path + "AgenticRAG/qa_generator.py", "QAGenerator"),
    "AtomicTaskGenerator": (cur_path + "AgenticRAG/atomic_task_generator.py", "AtomicTaskGenerator"),
    "DepthQAGenerator": (cur_path + "AgenticRAG/depth_qa_generator.py", "DepthQAGenerator"),
    "WidthQAGenerator": (cur_path + "AgenticRAG/width_qa_generator.py", "WidthQAGenerator"),
    
    # RARE
    "Doc2Query": (cur_path + "RARE/doc_to_query.py", "Doc2Query"),
    "BM25HardNeg": (cur_path + "RARE/bm25_hard_negative.py", "BM25HardNeg"),
    "ReasonDistill": (cur_path + "RARE/reason_distill.py", "ReasonDistill"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/generate/", _import_structure)
