from __future__ import annotations

import inspect
import math
from typing import Any, Generic, Mapping, Optional, Protocol, Type, ParamSpec

import pandas as pd

from dataflow.core.operator import OperatorABC
from dataflow.utils.storage import DataFlowStorage

from .memory_storage import InMemoryStorage


_INITP = ParamSpec("_INITP")
_RUNP = ParamSpec("_RUNP")


class _OperatorProto(Protocol[_INITP, _RUNP]):
    """Structural type that captures both ``__init__`` and ``run`` signatures.

    Pyright / Pylance infers ``_INITP`` and ``_RUNP`` from the concrete
    operator so that :meth:`op_cls_init` and :meth:`run` expose the
    original parameter lists for IDE auto-complete.
    """

    def __init__(self, *args: _INITP.args, **kwargs: _INITP.kwargs) -> None: ...

    def run(
        self,
        storage: DataFlowStorage,
        *args: _RUNP.args,
        **kwargs: _RUNP.kwargs,
    ) -> Any: ...


class _OpRunner:
    """Actor-side worker: each replica holds an independent operator instance.

    Receives logical record groups, runs each group through
    :class:`InMemoryStorage`, and returns one output group per input group.
    """

    def __init__(self, op_cls: type, op_init_args: tuple, op_init_kwargs: dict):
        self.op = op_cls(*op_init_args, **op_init_kwargs)

    def run(
        self,
        record_groups: list[list[dict]],
        run_params: list[dict],
    ) -> list[list[dict]]:
        outputs = []
        for records, params in zip(record_groups, run_params, strict=True):
            storage = InMemoryStorage(pd.DataFrame(records))
            self.op.run(storage, *params["args"], **params["kwargs"])
            outputs.append(storage.result.to_dict("records"))
        return outputs


def _build_operator_pipeline(
    op_cls: type,
    op_init_args: tuple,
    op_init_kwargs: dict,
    *,
    replicas: int,
    num_gpus_per_replica: float,
    runtime_env: Mapping[str, Any] | None,
):
    """Build the RayOrch pipeline only when the optional dependency is used."""

    from rayorch import Pipeline, RayModule

    class OperatorPipeline(Pipeline):
        def __init__(self) -> None:
            options: dict[str, Any] = {
                "replicas": replicas,
                "batch_size": 1,
                "num_gpus": num_gpus_per_replica,
            }
            if runtime_env is not None:
                options["runtime_env"] = dict(runtime_env)
            self.operator = (
                RayModule(_OpRunner)
                .pre_init(op_cls, op_init_args, op_init_kwargs)
                .ray_options(**options)
            )

        def forward(self, records, run_params):
            return self.operator(records, run_params)

    return OperatorPipeline()


class RayAcceleratedOperator(OperatorABC, Generic[_INITP, _RUNP]):
    """DataFlow operator backed by RayOrch for transparent data-parallel execution.

    From the pipeline's perspective this is a normal :class:`OperatorABC`:
    it reads from and writes to :class:`DataFlowStorage` sequentially.
    Internally it fans the DataFrame out to *replicas* Ray actors,
    each holding an independent copy of the wrapped operator (and its model).

    Actors are created **lazily** on the first ``run()`` call so that
    pipeline ``compile()`` does not trigger heavyweight model loading.

    Only suitable for **row-independent (map-style)** operators.  Operators
    that need cross-row global state (e.g. semantic dedup with a full
    similarity matrix) should *not* use this wrapper.

    Both ``op_cls_init`` and ``run`` have their signatures inferred from
    ``op_cls`` via ``ParamSpec``, giving full IDE auto-complete.

    Parameters
    ----------
    op_cls:
        The DataFlow operator class to parallelize.
    replicas:
        Number of parallel actor replicas.
    num_gpus_per_replica:
        Fractional GPU allocation per replica (e.g. ``0.25`` to share one
        GPU across four replicas).
    env:
        Backward-compatible shorthand for ``runtime_env={"conda": env}``.
    batch_size:
        Maximum rows passed to one actor call. By default each input is
        divided evenly across replicas, matching the previous contiguous-shard
        behavior.
    runtime_env:
        Optional Ray ``runtime_env`` forwarded to every actor.

    Example
    -------
    ::

        from dataflow.rayorch import RayAcceleratedOperator
        from dataflow.operators.text_pt.eval import FineWebEduSampleEvaluator

        scorer = RayAcceleratedOperator(
            FineWebEduSampleEvaluator,
            replicas=4,
            num_gpus_per_replica=0.25,
        ).op_cls_init(device="cuda")          # ← IDE shows __init__ params

        scorer.run(storage, input_key="text")  # ← IDE shows run params
    """

    def __init__(
        self,
        op_cls: Type[_OperatorProto[_INITP, _RUNP]],
        *,
        replicas: int = 1,
        num_gpus_per_replica: float = 0.0,
        env: Optional[str] = None,
        batch_size: int | None = None,
        runtime_env: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if type(replicas) is not int or replicas <= 0:
            raise ValueError("replicas must be a positive integer")
        if batch_size is not None and (
            type(batch_size) is not int or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer or None")
        if num_gpus_per_replica < 0:
            raise ValueError("num_gpus_per_replica cannot be negative")
        self._op_cls = op_cls
        self._op_init_args: tuple = ()
        self._op_init_kwargs: dict = {}
        self._replicas = replicas
        self._batch_size = batch_size
        self._num_gpus_per_replica = num_gpus_per_replica
        if env is not None and runtime_env is not None:
            raise ValueError("pass env or runtime_env, not both")
        self._runtime_env = (
            {"conda": env}
            if env is not None
            else None if runtime_env is None else dict(runtime_env)
        )
        self._executor = None  # Created lazily and reused across run() calls.

        # PipelineABC.compile() compatibility:
        # compile() → AutoOP uses inspect.signature(operator.run) to bind()
        # call arguments.  Our class-level run(storage, *args, **kwargs) would
        # cause bind() to dump extra params into *args, which later gets
        # serialised as an "args" key and leaks into the inner operator on
        # _compiled_forward replay.  Installing the inner operator's named
        # signature on the instance avoids this entirely.
        self._install_inner_run_signature(op_cls)

    def op_cls_init(
        self,
        *args: _INITP.args,
        **kwargs: _INITP.kwargs,
    ) -> RayAcceleratedOperator[_INITP, _RUNP]:
        """Configure how the wrapped operator is constructed inside each actor.

        Parameters match ``op_cls.__init__``, so IDE auto-complete works.
        May be omitted if the operator's defaults are sufficient.
        """
        self._op_init_args = args
        self._op_init_kwargs = kwargs
        return self

    def _ensure_initialized(self) -> None:
        if self._executor is not None:
            return
        from rayorch import Executor

        pipeline = _build_operator_pipeline(
            self._op_cls,
            self._op_init_args,
            self._op_init_kwargs,
            replicas=self._replicas,
            num_gpus_per_replica=self._num_gpus_per_replica,
            runtime_env=self._runtime_env,
        )
        # DataFlow may invoke one operator repeatedly, so keep the Executor
        # alive instead of using Pipeline.run(), which is intentionally one-shot.
        self._executor = Executor(pipeline)

    # --- inner signature propagation ---

    def _install_inner_run_signature(self, op_cls: type) -> None:
        """Replace ``self.run`` with a thin proxy carrying ``op_cls.run``'s
        ``__signature__``.

        Why: ``PipelineABC.compile()`` → ``AutoOP`` uses
        ``inspect.signature(operator.run)`` to ``bind()`` the call arguments.
        If the signature is the generic ``(storage, *args, **kwargs)`` from
        this wrapper, positional-overflow values land in ``*args`` and get
        serialised as an ``"args"`` key in the kwargs dict.  On replay via
        ``_compiled_forward(**kwargs)``, that ``"args"`` key leaks into the
        inner operator as an unexpected keyword argument.

        By exposing the inner operator's **named** parameters here,
        ``bind()`` resolves every argument to a keyword — no ``*args``
        residue, no downstream pollution.  Only this file changes; DataFlow
        core is untouched.
        """
        inner_sig = inspect.signature(op_cls.run)
        params = [p for p in inner_sig.parameters.values() if p.name != "self"]

        impl = self._run_impl

        def run(*args: Any, **kwargs: Any) -> None:
            return impl(*args, **kwargs)

        run.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
        run.__doc__ = getattr(op_cls.run, "__doc__", None)
        run.__name__ = "run"
        run.__qualname__ = f"{type(self).__qualname__}.run"
        self.run = run  # type: ignore[assignment]

    # --- DataFlow OperatorABC interface ---
    # Two-level design for compile() compatibility:
    #   1. Class-level `run` — satisfies OperatorABC's abstract method so the
    #      class can be instantiated.  Delegates to `_run_impl`.
    #   2. Instance-level `run` (proxy) — installed by
    #      `_install_inner_run_signature` in __init__, carries the inner
    #      operator's __signature__ so AutoOP.bind() resolves args to keywords.
    #      Python attribute lookup checks instance __dict__ before the class,
    #      so the proxy always wins at runtime.

    def run(  # type: ignore[override]
        self,
        storage: DataFlowStorage,
        *args: _RUNP.args,
        **kwargs: _RUNP.kwargs,
    ) -> None:
        return self._run_impl(storage, *args, **kwargs)

    def _run_impl(
        self,
        storage: DataFlowStorage,
        *args: _RUNP.args,
        **kwargs: _RUNP.kwargs,
    ) -> None:
        df = storage.read("dataframe")
        if df.empty:
            storage.write(df.copy())
            return
        records: list[dict] = df.to_dict("records")
        chunk_size = self._batch_size or max(
            1,
            math.ceil(len(records) / self._replicas),
        )
        record_groups = [
            records[start : start + chunk_size]
            for start in range(0, len(records), chunk_size)
        ]
        self._ensure_initialized()
        run_params: dict = {"args": args, "kwargs": kwargs}
        result = self._executor.run(
            record_groups,
            [run_params] * len(record_groups),
        )
        output_records = [
            record
            for group in result.outputs
            for record in group
        ]
        storage.write(pd.DataFrame(output_records))

    # --- lifecycle helpers ---

    def shutdown(self) -> None:
        """Terminate all Ray actors held by this operator."""
        if self._executor is None:
            return
        self._executor.close()
        self._executor = None

    def __repr__(self) -> str:
        state = "initialized" if self._executor is not None else "lazy"
        return (
            f"RayAcceleratedOperator({self._op_cls.__name__}, "
            f"replicas={self._replicas}, state={state})"
        )
