import sys
from dataflow.utils.registry import LazyLoader
from .Reasoning import *
from .GeneralText import *
from .AgenticRAG import *
cur_path = "dataflow/operators/process/"
_import_structure = {
    "AnswerFormatterFilter": (cur_path + "Reasoning/answer_formatter_filter.py", "AnswerFormatterFilter"),
    "AnswerGroundTruthFilter": (cur_path + "Reasoning/answer_groundtruth_filter.py", "AnswerGroundTruthFilter"),
    "AnswerJudger_Mathverify": (cur_path + "Reasoning/answer_judger_mathverify.py", "AnswerJudger_Mathverify"),
    "AnswerNgramFilter": (cur_path + "Reasoning/answer_ngram_filter.py", "AnswerNgramFilter"),
    "AnswerPipelineRoot": (cur_path + "Reasoning/answer_pipeline_root.py", "AnswerPipelineRoot"),
    "AnswerTokenLengthFilter": (cur_path + "Reasoning/answer_token_length_filter.py", "AnswerTokenLengthFilter"),
    "QuestionFilter": (cur_path + "Reasoning/question_filter.py", "QuestionFilter"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/process/", _import_structure)
