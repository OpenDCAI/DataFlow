from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # generate
    from .generate.answer_generator import AnswerGenerator
    from .generate.question_category_classifier import QuestionCategoryClassifier
    from .generate.question_difficulty_classifier import QuestionDifficultyClassifier
    from .generate.question_generator import QuestionGenerator
    from .generate.answer_extraction_qwenmatheval import AnswerExtraction_QwenMathEval
    from .generate.pseudo_answer_generator import PseudoAnswerGenerator
    from .generate.pretrain_format_converter import PretrainFormatConverter

    # eval
    from .eval.category_info import CategoryInfo
    from .eval.difficulty_info import DifficultyInfo
    from .eval.token_info import ToKenInfo
    
    # filter
    from .filter.answer_formatter_filter import AnswerFormatterFilter
    from .filter.answer_groundtruth_filter import AnswerGroundTruthFilter
    from .filter.answer_judger_mathverify import AnswerJudger_MathVerify
    from .filter.answer_ngram_filter import AnswerNgramFilter
    from .filter.answer_pipeline_root import AnswerPipelineRoot
    from .filter.answer_token_length_filter import AnswerTokenLengthFilter
    from .filter.question_filter import QuestionFilter
    from .filter.answer_model_judge import AnswerModelJudge

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/reasoning/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/reasoning/", _import_structure)
