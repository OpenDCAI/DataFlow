import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "ClipScorer": ("dataflow/Eval/image/clip_scorer.py", "ClipScorer"),
    "LongClipScorer": ("dataflow/Eval/image/longclip_scorer.py", "LongClipScorer"),
    "ClipT5Scorer": ("dataflow/Eval/image/clip_t5_scorer.py", "ClipT5Scorer"),
    "FleurScorer": ("dataflow/Eval/image/fleur_scorer.py", "FleurScorer"),
    "FIDScorer": ("dataflow/Eval/image/fid_scorer.py", "FIDScorer"),
    "KIDScorer": ("dataflow/Eval/image/kid_scorer.py", "KIDScorer"),
    "ISScorer": ("dataflow/Eval/image/is_scorer.py", "ISScorer"),
    "ImageResolutionScorer": ("dataflow/Eval/image/image_resolution_scorer.py", "ImageResolutionScorer"),
    "ImageAspectRatioScorer": ("dataflow/Eval/image/image_aspect_ratio_scorer.py", "ImageAspectRatioScorer"),
    "QalignScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "QalignScorer"),
    "LiqeScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "LiqeScorer"),
    "ArniqaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "ArniqaScorer"),
    "TopiqScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "TopiqScorer"),
    "ClipiqaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "ClipiqaScorer"),
    "ManiqaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "ManiqaScorer"),
    "MusiqScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "MusiqScorer"),
    "DbcnnScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "DbcnnScorer"),
    "Pqa2piqScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "Pqa2piqScorer"),
    "HyperiqaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "HyperiqaScorer"),
    "NimaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "NimaScorer"),
    "WadiqamScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "WadiqamScorer"),
    "CnniqaScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "CnniqaScorer"),
    "NrqmScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "NrqmScorer"),
    "PiScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "PiScorer"),
    "BrisqueScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "BrisqueScorer"),
    "IlniqeScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "IlniqeScorer"),
    "NiqeScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "NiqeScorer"),
    "PiqeScorer": ("dataflow/Eval/image/pyiqa_scorer.py", "PiqeScorer"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/Eval/image", _import_structure)

# import importlib

# # from .clip_scorer import ClipScorer
# # from .longclip_scorer import LongClipScorer
# # from .clip_t5_scorer import ClipT5Scorer
# # from .fleur_scorer import FleurScorer
# from .image_text_scorer import *
# from .pyiqa_scorer import *
# from .fid_scorer import FIDScorer
# from .kid_scorer import KIDScorer
# from .is_scorer import ISScorer

# from .image_resolution_scorer import ImageResolutionScorer
# from .image_aspect_ratio_scorer import ImageAspectRatioScorer

# pyiqa_module = importlib.import_module("dataflow.Eval.image.pyiqa_scorer")
# pyiqa_scorer = [attr_name for attr_name in dir(pyiqa_module) if attr_name.endswith("Scorer") and attr_name not in ["PyiqaScorer", "ImageScorer"]]
# pure_image_scorer_class = pyiqa_module
# pure_image_scorer_name = pyiqa_scorer
# image_text_scorer_class = importlib.import_module("dataflow.Eval.image.image_text_scorer")
# image_text_scorer_name = [attr_name for attr_name in dir(image_text_scorer_class) if attr_name.endswith("Scorer")]

# # __all__ = pure_image_scorer_name + image_text_scorer_name + [
# #     "FIDScorer",
# #     "KIDScorer",
# #     "ISScorer",
# #     "ImageResolutionScorer",
# #     "ImageAspectRatioScorer",
# # ]

# # __all__ = [
# #     'clipModel',
# #     'pyiqaModel',
# #     'scorer',
# #     'scorerModel'
# # ]

# # pure_image = [
# #     "QalignScorer",
# #     "LiqeScorer",
# #     "ArniqaScorer",
# #     "TopiqScorer",
# #     "ClipiqaScorer",
# #     "ManiqaScorer",
# #     "MusiqScorer",
# #     "DbcnnScorer",
# #     "Pqa2piqScorer",
# #     "HyperiqaScorer",
# #     "NimaScorer",
# #     "WadiqamScoreer",
# #     "CnniqaScorer",
# #     "NrqmScoreer",
# #     "PiScorer",
# #     "BrisqueScorer",
# #     "IlniqeScorer",
# #     "NiqeScorer",
# #     "PiqeScorer",
# # ]