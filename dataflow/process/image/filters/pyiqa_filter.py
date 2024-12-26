import numpy as np

from dataflow.core import ImageFilter
# from dataflow.Eval.image import pure_image_scorer_class, pure_image_scorer_name
from dataflow.utils.registry import PROCESSOR_REGISTRY

TYPE_KEY = "type"

class IQAFilter(ImageFilter):
    def __init__(self, scorer, args_dict: dict):
        super().__init__()
        self.min_score = args_dict["min_score"] if "min_score" in args_dict else -np.inf
        self.max_score = args_dict["max_score"] if "max_score" in args_dict else np.inf
        
        self.scorer = scorer(args_dict=args_dict)
        self.metric_type = args_dict[TYPE_KEY] if TYPE_KEY in args_dict else "Default"

    def filter_func(self, sample):
        _, score = self.scorer(sample)

        result = np.array(((self.min_score <= score[self.metric_type]) & (score[self.metric_type] <= self.max_score)).astype(int))
        return result

@PROCESSOR_REGISTRY.register()
class QalignFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import QalignScorer
        super().__init__(QalignScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class LiqeFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import LiqeScorer
        super().__init__(LiqeScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class ArniqaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import ArniqaScorer
        super().__init__(ArniqaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class TopiqFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import TopiqScorer
        super().__init__(TopiqScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class ClipiqaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import ClipiqaScorer
        super().__init__(ClipiqaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class ManiqaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import ManiqaScorer
        super().__init__(ManiqaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class MusiqFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import MusiqScorer
        super().__init__(MusiqScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class DbcnnFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import DbcnnScorer
        super().__init__(DbcnnScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class Pqa2piqFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import Pqa2piqScorer
        super().__init__(Pqa2piqScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class HyperiqaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import HyperiqaScorer
        super().__init__(HyperiqaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class NimaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import NimaScorer
        super().__init__(NimaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class WadiqamFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import WadiqamScorer
        super().__init__(WadiqamScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class CnniqaFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import CnniqaScorer
        super().__init__(CnniqaScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class NrqmFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import NrqmScorer
        super().__init__(NrqmScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class PiFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import PiScorer
        super().__init__(PiScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class BrisqueFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import BrisqueScorer
        super().__init__(BrisqueScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class IlniqeFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import IlniqeScorer
        super().__init__(IlniqeScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class NiqeFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import NiqeScorer
        super().__init__(NiqeScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class PiqeFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import PiqeScorer
        super().__init__(PiqeScorer, args_dict)



# def create_pyiqa_filter_classes():
#     for attr_name in pure_image_scorer_name:
#         scorer_class = getattr(pure_image_scorer_class, attr_name)
#         filter_class_name = attr_name.replace("Scorer", "Filter")

#         def init_method(self, args_dict, scorer_class=scorer_class):
#             IQAFilter.__init__(self, scorer_class, args_dict)
        
#         new_class = type(
#             filter_class_name,
#             (IQAFilter,),
#             {
#                 '__init__': init_method,
#             }
#         )
        
#         PROCESSOR_REGISTRY.register(new_class)


# create_pyiqa_filter_classes()