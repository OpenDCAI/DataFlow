import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "VideoResolutionFilter": ("dataflow/process/video/filters/video_resolution_filter.py", "VideoResolutionFilter"),
    "VideoMotionFilter": ("dataflow/process/video/filters/video_motion_filter.py", "VideoMotionFilter"),
    "FastVQAFilter": ("dataflow/process/video/filters/fastvqa_filter.py", "FastVQAFilter"),
    "FasterVQAFilter": ("dataflow/process/video/filters/fastervqa_filter.py", "FasterVQAFilter"),
    "DOVERFilter": ("dataflow/process/video/filters/dover_filter.py", "DOVERFilter"),
    "EMScoreFilter": ("dataflow/process/video/filters/emscore_filter.py", "EMScoreFilter"),
    "PACScoreFilter": ("dataflow/process/video/filters/pacscore_filter.py", "PACScoreFilter")
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/video/filters", _import_structure)
