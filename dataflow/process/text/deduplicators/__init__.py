import sys
from dataflow.utils.registry import LazyLoader

_import_structure = {
    "HashDeduplicator": ("dataflow/process/text/deduplicators/hash_deduplicator.py", "HashDeduplicator"),
    "SemDeduplicator": ("dataflow/process/text/deduplicators/sem_deduplicator.py", "SemDeduplicator"),
    "SimHashDeduplicator": ("dataflow/process/text/deduplicators/simhash_deduplicator.py", "SimHashDeduplicator"),
    "CCNetDeduplicator": ("dataflow/process/text/deduplicators/ccnet_deduplicator.py", "CCNetDeduplicator"),
    "NgramHashDeduplicator": ("dataflow/process/text/deduplicators/ngramhash_deduplicator.py", "NgramHashDeduplicator"),
    "MinHashDeduplicator": ("dataflow/process/text/deduplicators/minhash_deduplicator.py", "MinHashDeduplicator")
}

sys.modules[__name__] = LazyLoader(__name__, "dataflow/process/text/deduplicators", _import_structure)

