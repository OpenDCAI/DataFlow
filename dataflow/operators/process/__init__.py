import sys
from dataflow.utils.registry import LazyLoader
from .Reasoning import *
# from .GeneralText import *
from .AgenticRAG import *
cur_path = "dataflow/operators/process/"
_import_structure = {
    "AnswerFormatterFilter": (cur_path + "Reasoning/AnswerFormatterFilter.py", "AnswerFormatterFilter"),
    "AnswerGroundTruthFilter": (cur_path + "Reasoning/AnswerGroundTruthFilter.py", "AnswerGroundTruthFilter"),
    "AnswerJudger_Mathverify": (cur_path + "Reasoning/AnswerJudger_Mathverify.py", "AnswerJudger_Mathverify"),
    "AnswerNgramFilter": (cur_path + "Reasoning/AnswerNgramFilter.py", "AnswerNgramFilter"),
    "AnswerPipelineRoot": (cur_path + "Reasoning/AnswerPipelineRoot.py", "AnswerPipelineRoot"),
    "AnswerTokenLengthFilter": (cur_path + "Reasoning/AnswerTokenLengthFilter.py", "AnswerTokenLengthFilter"),
    "QuestionFilter": (cur_path + "Reasoning/QuestionFilter.py", "QuestionFilter"),
    'ContentChooser': (cur_path + "AgenticRAG/ContentChooser.py", "ContentChooser"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/process/", _import_structure)
