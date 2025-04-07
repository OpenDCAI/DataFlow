import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "MathProblemFilter": ("dataflow/process/text/reasoners/math_problem_filter.py", "MathProblemFilter"),
    "AnswerGroundTruthFilter": ("dataflow/process/text/reasoners/answer_ground_truth_filter.py", "AnswerGroundTruthFilter"),
    "AnswerFormatterFilter": ("dataflow/process/text/reasoners/answer_formatter_filter.py", "AnswerFormatterFilter"),
    "AnswerNgramFilter": ("dataflow/process/text/reasoners/answer_ngram_filter.py", "AnswerNgramFilter"),
    "AnswerTokenLengthFilter": ("dataflow/process/text/reasoners/answer_token_length_filter.py", "AnswerTokenLengthFilter"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/text/reasoners", _import_structure)