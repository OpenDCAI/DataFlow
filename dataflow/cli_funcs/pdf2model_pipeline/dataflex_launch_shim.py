#!/usr/bin/env python3
"""Subprocess entry for DataFlow → DataFlex training (pdf2model and future backends).
"""

from __future__ import annotations

import inspect


def apply_accelerate_unwrap_compat_patch() -> None:
    from accelerate import Accelerator

    orig = Accelerator.unwrap_model

    def unwrap_model(self, model, *args, **kwargs):
        kwargs = dict(kwargs)
        try:
            params = inspect.signature(orig).parameters
            if "keep_torch_compile" in kwargs and "keep_torch_compile" not in params:
                kwargs.pop("keep_torch_compile", None)
        except (TypeError, ValueError):
            kwargs.pop("keep_torch_compile", None)
        return orig(self, model, *args, **kwargs)

    try:
        params = inspect.signature(orig).parameters
        if "keep_torch_compile" not in params:
            Accelerator.unwrap_model = unwrap_model  # type: ignore[method-assign]
    except (TypeError, ValueError):
        Accelerator.unwrap_model = unwrap_model  # type: ignore[method-assign]


def apply_transformers_seed_worker_compat_patch() -> None:
    import transformers.trainer_utils as tu

    _orig = tu.seed_worker
    try:
        params = list(inspect.signature(_orig).parameters)
        if len(params) <= 1:
            return
    except (TypeError, ValueError):
        return

    def seed_worker(worker_id: int) -> None:
        import torch
        import torch.distributed as dist

        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info is not None else 0
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        return _orig(worker_id, num_workers=num_workers, rank=rank)

    tu.seed_worker = seed_worker  # type: ignore[assignment]


def main() -> None:
    apply_accelerate_unwrap_compat_patch()
    apply_transformers_seed_worker_compat_patch()
    from dataflex.launcher import launch

    launch()


if __name__ == "__main__":
    main()
