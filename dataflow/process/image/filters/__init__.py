# from .image_aspect_ratio_filter import ImageAspectRatioFilter
# from .image_resolution_filter import ImageResolutionFilter
# # from . import pyiqa_filter 
# from .pyiqa_filter import * 
# from . import image_text_similarity_filter

# __all__ = [
#     'ImageResolutionFilter',
#     'ImageAspectRatioFilter',
# ]

import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "QalignFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "QalignFilter"),
    "LiqeFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "LiqeFilter"),
    "ArniqaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "ArniqaFilter"),
    "TopiqFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "TopiqFilter"),
    "ClipiqaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "ClipiqaFilter"),
    "ManiqaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "ManiqaFilter"),
    "MusiqFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "MusiqFilter"),
    "DbcnnFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "DbcnnFilter"),
    "Pqa2piqFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "Pqa2piqFilter"),
    "HyperiqaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "HyperiqaFilter"),
    "NimaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "NimaFilter"),
    "WadiqamFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "WadiqamFilter"),
    "CnniqaFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "CnniqaFilter"),
    "NrqmFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "NrqmFilter"),
    "PiFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "PiFilter"),
    "BrisqueFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "BrisqueFilter"),
    "IlniqeFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "IlniqeFilter"),
    "NiqeFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "NiqeFilter"),
    "PiqeFilter": ("dataflow/process/image/filters/pyiqa_filter.py", "PiqeFilter"),
    "ImageResolutionFilter": ("dataflow/process/image/filters/image_resolution_filter.py", "ImageResolutionFilter"),
    "ImageAspectRatioFilter": ("dataflow/process/image/filters/image_aspect_ratio_filter.py", "ImageAspectRatioFilter"),
    "ClipFilter": ("dataflow/process/image/filters/image_text_similarity_filter.py", "ClipFilter"),
    "LongClipFilter": ("dataflow/process/image/filters/image_text_similarity_filter.py", "LongClipFilter"),
    "ClipT5Filter": ("dataflow/process/image/filters/image_text_similarity_filter.py", "ClipT5Filter"),
    "FleurFilter": ("dataflow/process/image/filters/image_text_similarity_filter.py", "FleurFilter"),
}
sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/image/filter", _import_structure)