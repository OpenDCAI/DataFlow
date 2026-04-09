"""RayOrch integration for DataFlow — transparent data-parallel acceleration.

Usage::

    from dataflow.rayorch import RayAcceleratedOperator

    scorer = RayAcceleratedOperator(
        FineWebEduSampleEvaluator,
        replicas=4,
        num_gpus_per_replica=0.25,
    ).op_cls_init(device="cuda")
    scorer.run(storage, input_key="text", output_key="edu_score")
"""

from .accelerated_op import RayAcceleratedOperator
from .memory_storage import InMemoryStorage as _InMemoryStorage

__all__ = [
    "RayAcceleratedOperator",
]
