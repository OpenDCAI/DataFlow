import sys
from dataflow.utils.registry import LazyLoader

cur_path = "dataflow/generator/algorithms/"
_import_structure = {
    "AnswerExtraction_qwenmatheval": (cur_path + "AnswerExtraction_qwenmatheval" + ".py", "AnswerExtraction_qwenmatheval"),
    "AnswerGenerator_reasoning": (cur_path + "AnswerGenerator_reasoning" + ".py", "AnswerGenerator_reasoning"),
    "AnswerGenerator": (cur_path + "AnswerGenerator" + ".py", "AnswerGenerator"),
    "AnswerJudger_mathverify": (cur_path + "AnswerJudger_mathverify" + ".py", "AnswerJudger_mathverify"),
    "AnswerJudger_xverify": (cur_path + "AnswerJudger_xverify" + ".py", "AnswerJudger_xverify"),
    "AnswerPipelineRoot": (cur_path + "AnswerPipelineRoot" + ".py", "AnswerPipelineRoot"),
    "CodeCommentGenerator": (cur_path + "CodeCommentGenerator" + ".py", "CodeCommentGenerator"),
    "CodeFilter": (cur_path + "CodeFilter" + ".py", "CodeFilter"),
    "CodeRefiner": (cur_path + "CodeRefiner" + ".py", "CodeRefiner"),
    "CodeScorer": (cur_path + "CodeScorer" + ".py", "CodeScorer"),
    "DatabaseSchemaExtractor": (cur_path + "DatabaseSchemaExtractor" + ".py", "DatabaseSchemaExtractor"),
    "ExtractLines": (cur_path + "ExtractLines" + ".py", "ExtractLines"),
    "ExtraKnowledgeGenerator": (cur_path + "ExtraKnowledgeGenerator" + ".py", "ExtraKnowledgeGenerator"),
    "LanguageClassifier": (cur_path + "LanguageClassifier" + ".py", "LanguageClassifier"),
    "MCTSAnswerGenerator": (cur_path + "MCTSAnswerGenerator" + ".py", "MCTSAnswerGenerator"),
    "OSSInstGenerator": (cur_path + "OSSInstGenerator" + ".py", "OSSInstGenerator"),
    "PromptGenerator": (cur_path + "PromptGenerator" + ".py", "PromptGenerator"),
    "PseudoAnswerGenerator_reasoning": (cur_path + "PseudoAnswerGenerator_reasoning" + ".py", "PseudoAnswerGenerator_reasoning"),
    "PseudoAnswerGenerator": (cur_path + "PseudoAnswerGenerator" + ".py", "PseudoAnswerGenerator"),
    "QuestionCategoryClassifier": (cur_path + "QuestionCategoryClassifier" + ".py", "QuestionCategoryClassifier"),
    "QuestionDifficultyClassifier": (cur_path + "QuestionDifficultyClassifier" + ".py", "QuestionDifficultyClassifier"),
    "QuestionGenerator": (cur_path + "QuestionGenerator" + ".py", "QuestionGenerator"),
    "QuestionRefiner": (cur_path + "QuestionRefiner" + ".py", "QuestionRefiner"),
    "SchemaLinking": (cur_path + "SchemaLinking" + ".py", "SchemaLinking"),
    "SQLDifficultyClassifier": (cur_path + "SQLDifficultyClassifier" + ".py", "SQLDifficultyClassifier"),
    "SQLFilter": (cur_path + "SQLFilter" + ".py", "SQLFilter"),
    "StaticCodeChecker": (cur_path + "StaticCodeChecker" + ".py", "StaticCodeChecker"),
    "Text2SQLDifficultyClassifier": (cur_path + "Text2SQLDifficultyClassifier" + ".py", "Text2SQLDifficultyClassifier"),
    "TextSQLConsistency": (cur_path + "TextSQLConsistency" + ".py", "TextSQLConsistency"),
    "TreeSitterParser": (cur_path + "TreeSitterParser" + ".py", "TreeSitterParser"),
    "PretrainGenerator": (cur_path + "PretrainGenerator" + ".py", "PretrainGenerator"),
    "SupervisedFinetuneGenerator": (cur_path + "SupervisedFinetuneGenerator" + ".py", "SupervisedFinetuneGenerator"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/generator/algorithm", _import_structure)
