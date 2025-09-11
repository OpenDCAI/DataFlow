from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # generate
    from .generate.reasoning_answer_generator import ReasoningAnswerGenerator
    from .generate.reasoning_question_generator import ReasoningQuestionGenerator
    from .generate.reasoning_answer_extraction_qwenmatheval_generator import ReasoningAnswerExtractionQwenMathEvalGenerator
    from .generate.reasoning_pseudo_answer_generator import ReasoningPseudoAnswerGenerator
    from .generate.reasoning_pretrain_format_convert_generator import ReasoningPretrainFormatConvertGenerator

    # eval
    from .eval.reasoning_category_dataset_evaluator import ReasoningCategoryDatasetEvaluator
    from .eval.reasoning_difficulty_dataset_evaluator import ReasoningDifficultyDatasetEvaluator
    from .eval.reasoning_token_dataset_evaluator import ReasoningTokenDatasetEvaluator
    from .eval.reasoning_question_category_sample_evaluator import ReasoningQuestionCategorySampleEvaluator
    from .eval.reasoning_question_difficulty_sample_evaluator import ReasoningQuestionDifficultySampleEvaluator
    
    # filter
    from .filter.reasoning_answer_formatter_filter import ReasoningAnswerFormatterFilter
    from .filter.reasoning_answer_groundtruth_filter import ReasoningAnswerGroundTruthFilter
    from .filter.reasoning_answer_judge_mathverify_filter import ReasoningAnswerJudgeMathVerifyFilter
    from .filter.reasoning_answer_ngram_filter import ReasoningAnswerNgramFilter
    from .filter.reasoning_answer_pipeline_root_filter import ReasoningAnswerPipelineRootFilter
    from .filter.reasoning_answer_token_length_filter import ReasoningAnswerTokenLengthFilter
    from .filter.reasoning_question_filter import ReasoningQuestionFilter
    from .filter.reasoning_answer_model_judge_filter import ReasoningAnswerModelJudgeFilter

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/reasoning/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/reasoning/", _import_structure)
