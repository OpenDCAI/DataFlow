import numpy as np

from dataflow.core import ImageTextFilter
# from dataflow.Eval.image import image_text_scorer_class, image_text_scorer_name
from dataflow.utils.registry import PROCESSOR_REGISTRY

TYPE_KEY = "type"

class IQAFilter(ImageTextFilter):
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
class ClipFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import ClipScorer
        super().__init__(ClipScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class LongClipFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import LongClipScorer
        super().__init__(LongClipScorer, args_dict)

@PROCESSOR_REGISTRY.register()
class ClipT5Filter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import ClipT5Scorer
        super().__init__(ClipT5Scorer, args_dict)

@PROCESSOR_REGISTRY.register()
class FleurFilter(IQAFilter):
    def __init__(self, args_dict):
        from dataflow.Eval.image import FleurScorer
        super().__init__(FleurScorer, args_dict)



# def create_image_text_filter_classes():
#     for attr_name in image_text_scorer_name:
#         scorer_class = getattr(image_text_scorer_class, attr_name)
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

# create_image_text_filter_classes()