import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "ImagePHashDeduplicator": ("dataflow/process/image/deduplicators/deduplicator.py", "ImagePHashDeduplicator"),
    "ImageAHashDeduplicator": ("dataflow/process/image/deduplicators/deduplicator.py", "ImageAHashDeduplicator"),
    "ImageDHashDeduplicator": ("dataflow/process/image/deduplicators/deduplicator.py", "ImageDHashDeduplicator"),
    "ImageWHashDeduplicator": ("dataflow/process/image/deduplicators/deduplicator.py", "ImageWHashDeduplicator"),
}
sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/image/deduplicators/", _import_structure)

# from .deduplicator import ImagePHashDeduplicator, ImageAHashDeduplicator, ImageDHashDeduplicator, ImageWHashDeduplicator
