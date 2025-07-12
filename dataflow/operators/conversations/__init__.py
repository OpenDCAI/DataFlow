import sys
from dataflow.utils.registry import LazyLoader
from .consistent_chat import ConsistentChatGenerator

cur_path = "dataflow/operators/conversations/"

_import_structure = {
    "ConsistentChatGenerator": (cur_path + "consistent_chat.py", "ConsistentChatGenerator"),
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/operators/conversations/", _import_structure)