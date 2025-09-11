from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .eval.alpagasus_sample_evaluator import AlpagasusSampleEvaluator
    from .eval.deita_quality_sample_evaluator import DeitaQualitySampleEvaluator
    from .eval.deita_complexity_sample_evaluator import DeitaComplexitySampleEvaluator
    from .eval.instag_sample_evaluator import InstagSampleEvaluator
    from .eval.rm_sample_evaluator import RMSampleEvaluator
    from .eval.superfiltering_sample_evaluator import SuperfilteringSampleEvaluator
    from .eval.treeinstruct_sample_evaluator import TreeinstructSampleEvaluator


    from .filter.alpagasus_filter import AlpagasusFilter
    from .filter.deita_quality_filter import DeitaQualityFilter
    from .filter.deita_complexity_filter import DeitaComplexityFilter
    from .filter.instag_filter import InstagFilter
    from .filter.rm_filter import RMFilter
    from .filter.superfiltering_filter import SuperfilteringFilter
    from .filter.treeinstruct_filter import TreeinstructFilter
    
    from .generate.condor_generator import CondorGenerator
    from .generate.sft_generator_from_seed import SFTGeneratorSeed
    
    from .refine.condor_refiner import CondorRefiner
    
else:
    import sys
    from dataflow.utils.registry import LazyLoader, generate_import_structure_from_type_checking

    cur_path = "dataflow/operators/text_sft/"

    _import_structure = generate_import_structure_from_type_checking(__file__, cur_path)
    sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/text_sft/", _import_structure)
