# import importlib
# from os import path as osp
# import os

# # from dataflow.utils.registry import MODEL_REGISTRY

# # automatically scan and import arch modules for registry
# folder = osp.dirname(osp.abspath(__file__))
# filenames = [f[:-3] for f in os.listdir(folder) if f.endswith('.py') and f != '__init__.py']
# _arch_modules = [importlib.import_module(f'dataflow.Eval.image.{file_name}') for file_name in filenames]


from .clip_scorer import ClipScorer
from .longclip_scorer import LongClipScorer
from .pyiqa_scorer import PyiqaScorer
from .clip_t5_scorer import ClipT5Scorer
from .fleur_scorer import FleurScorer
from .pyiqa_scorer import PyiqaScorer
from .fid_scorer import FIDScorer
from .kid_scorer import KIDScorer
from .is_scorer import ISScorer


# __all__ = [
#     'clipModel',
#     'pyiqaModel',
#     'scorer',
#     'scorerModel'
# ]