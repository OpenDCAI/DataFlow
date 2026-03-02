from .utils import *
from .version import __version__, version_info
from .logger import get_logger
from .operators import *
from .prompts import *
__all__ = [
    '__version__',
    'version_info',
    'get_logger',
]


def get_version():
    """Return the package version string."""
    return __version__


def hello():
    return "Hello from open-dataflow!"