"""Shared execution loop for compiled batch pipelines."""

from __future__ import annotations

import os
from pathlib import Path

from tqdm import tqdm


def _automatic_checkpoint_path(pipeline, batch_size):
    """Return a private manifest path when the storage contract is safe."""
    from dataflow.utils.storage import BatchedFileStorage

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        return None
    nodes = [node for node in pipeline.op_nodes_list if node.op_obj is not None]
    if not nodes or any(
        not isinstance(node.storage, BatchedFileStorage)
        or node.storage.cache_type != "jsonl"
        for node in nodes
    ):
        return None
    source = Path(nodes[0].storage.first_entry_file_name).resolve()
    if not source.is_file() or source.suffix.lower() != ".jsonl":
        return None
    if any(Path(node.storage.first_entry_file_name).resolve() != source for node in nodes):
        return None

    directory = Path(nodes[0].storage.cache_path).resolve() / ".dataflow_checkpoint"
    manifest = directory / "manifest.json"
    output_paths = [
        Path(node.storage._get_cache_file_path(node.storage.operator_step + 1)).resolve()
        for node in nodes
    ]
    if not manifest.exists() and any(path.exists() for path in output_paths):
        return None
    return directory


def _run_operator(node, frame=None, batch_step=None):
    storage = node.storage
    if batch_step is not None:
        storage.batch_step = batch_step
    if frame is not None:
        storage._active_batch = frame
    try:
        node.op_obj.run(storage=storage, **node.kwargs)
    finally:
        if frame is not None:
            storage._active_batch = None


def run_legacy_pipeline(pipeline, resume_step, batch_size, resume_from_last):
    """Run the compatibility cursor through one shared batch loop.

    The historical cursor format remains supported for existing callers. New
    crash-safe runs should use ``checkpoint_dir`` and the manifest runner.
    """
    if resume_step > 0 and resume_from_last:
        raise ValueError("Cannot set both `resume_step` and `resume_from_last` to True.")

    resume_batch = 0
    cache_path = None
    if resume_from_last:
        cache_path = os.path.join(
            pipeline.op_nodes_list[1].storage.cache_path,
            f"{pipeline.op_nodes_list[1].storage.file_name_prefix}_last_success_step.txt",
        )
        if not os.path.exists(cache_path):
            resume_step = 0
            pipeline.logger.info(
                f"No last success step cache found at {cache_path}, starting from step 0."
            )
        else:
            with open(cache_path, "r") as stream:
                resume_step, resume_batch = map(int, stream.readline().strip().split(","))
            pipeline.logger.info(
                f"Resuming from last success step {resume_step}, batch step {resume_batch}."
            )

    for index, node in enumerate(pipeline.op_nodes_list):
        step = index - 1
        if step < resume_step:
            continue

        if node.llm_serving is not None:
            if pipeline.active_llm_serving and pipeline.active_llm_serving is not node.llm_serving:
                pipeline.active_llm_serving.cleanup()
            pipeline.active_llm_serving = node.llm_serving

        if node.op_obj is not None:
            storage = node.storage
            start = resume_batch if step == resume_step else 0
            if batch_size is None:
                batches = ((0, None),)
            else:
                storage.batch_size = batch_size
                batches = enumerate(storage.iter_batches())

            for batch, frame in tqdm(
                batches,
                desc=f"\033[1;36mRunning {node.op_name} with batch size={batch_size}\033[0m",
                position=0,
                dynamic_ncols=True,
                colour="cyan",
            ):
                if batch < start:
                    continue
                _run_operator(
                    node,
                    frame=frame,
                    batch_step=batch if batch_size is not None else None,
                )
                if resume_from_last:
                    resume_batch = batch + 1 if batch_size is not None else 0
                    with open(cache_path, "w") as stream:
                        stream.write(f"{step},{resume_batch}\n")

            if hasattr(node.op_obj, "shutdown"):
                pipeline.logger.debug(
                    f"Auto-shutting down {node.op_name} to release actor resources."
                )
                node.op_obj.shutdown()

        if resume_from_last:
            resume_batch = 0
            with open(cache_path, "w") as stream:
                stream.write(f"{index},{resume_batch}\n")

        if node.llm_serving is not None:
            pipeline.llm_serving_counter[pipeline.active_llm_serving] -= 1
            if pipeline.llm_serving_counter[pipeline.active_llm_serving] == 0:
                pipeline.active_llm_serving.cleanup()
                pipeline.active_llm_serving = None


def run_compiled_pipeline(pipeline, resume_step, batch_size, resume_from_last):
    """Run a compiled pipeline with one recovery model when possible.

    New local JSONL runs use the transactional manifest automatically when the
    caller uses the historical ``resume_from_last=True`` default. Other
    backends retain their compatibility behavior until they provide the same
    storage contract.
    """
    if resume_from_last and resume_step == 0:
        directory = _automatic_checkpoint_path(pipeline, batch_size)
        if directory is not None:
            from dataflow.utils.pipeline_checkpoint import run_checkpointed_pipeline

            return run_checkpointed_pipeline(
                pipeline,
                directory,
                batch_size,
                {"compatibility": "resume_from_last"},
            )
    return run_legacy_pipeline(pipeline, resume_step, batch_size, resume_from_last)

