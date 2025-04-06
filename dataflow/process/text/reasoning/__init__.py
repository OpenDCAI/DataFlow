import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "AnswerGroundTruthFilter": ("dataflow/process/text/reasoning/answer_ground_truth_filter.py", "AnswerGroundTruthFilter"),
    "AnswerFormatterFilter": ("dataflow/process/text/reasoning/answer_formatter_filter.py", "AnswerFormatterFilter"),
    "AnswerNgramFilter": ("dataflow/process/text/reasoning/answer_ngram_filter.py", "AnswerNgramFilter"),
    "AnswerTokenLengthFilter": ("dataflow/process/text/reasoning/answer_token_length_filter.py", "AnswerTokenLengthFilter"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/text/reasoning", _import_structure)

