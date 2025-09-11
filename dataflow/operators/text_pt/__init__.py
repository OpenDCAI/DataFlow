from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # filter
    from .filter.ccnet_deduplicate_filter import CCNetDeduplicateFilter
    from .filter.debertav3_filter import DebertaV3Filter
    from .filter.fineweb_edu_filter import FineWebEduFilter
    from .filter.pair_qual_filter import PairQualFilter
    from .filter.perplexity_filter import PerplexityFilter
    from .filter.qurating_filter import QuratingFilter
    from .filter.text_book_filter import TextbookFilter

    # generate
    from .generate.phi4qa_generator import Phi4QAGenerator
    
    # eval
    from .eval.debertav3_sample_evaluator import DebertaV3SampleEvaluator
    from .eval.fineweb_edu_sample_evaluator import FineWebEduSampleEvaluator
    from .eval.pair_qual_sample_evaluator import PairQualSampleEvaluator
    from .eval.textbook_sample_evaluator import TextbookSampleEvaluator
    from .eval.qurating_sample_evaluator import QuratingSampleEvaluator
    from .eval.perplexity_sample_evaluator import PerplexitySampleEvaluator
    from .eval.meta_sample_evaluator import MetaSampleEvaluator

else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/text_pt/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/text_pt/", _import_structure)
