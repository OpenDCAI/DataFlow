from __future__ import annotations

import inspect
from typing import Any, Generic, Optional, Protocol, Type, ParamSpec

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

    Receives a chunk of records (``list[dict]``), wraps it in
    :class:`InMemoryStorage`, delegates to the DataFlow operator's ``run``,
    and returns the result as ``list[dict]``.
    """

    def __init__(self, op_cls: type, op_init_args: tuple, op_init_kwargs: dict):
        self.op = op_cls(*op_init_args, **op_init_kwargs)

    def run(self, records: list[dict], run_params: dict) -> list[dict]:
        if not records:
            return []
        df = pd.DataFrame(records)
        storage = InMemoryStorage(df)
        self.op.run(storage, *run_params.get("args", ()), **run_params.get("kwargs", {}))
        return storage.result.to_dict("records")


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
        Optional RayOrch ``EnvRegistry`` key for a custom ``runtime_env``.

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
    ):
        super().__init__()
        self._op_cls = op_cls
        self._op_init_args: tuple = ()
        self._op_init_kwargs: dict = {}
        self._replicas = replicas
        self._num_gpus_per_replica = num_gpus_per_replica
        self._env = env
        self._module = None  # created lazily

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
        if self._module is not None:
            return
        from rayorch import Dispatch, RayModule

        self._module = RayModule(
            _OpRunner,
            replicas=self._replicas,
            num_gpus_per_replica=self._num_gpus_per_replica,
            dispatch_mode=Dispatch.SHARD_CONTIGUOUS,
            env=self._env,
        )
        self._module.pre_init(
            op_cls=self._op_cls,
            op_init_args=self._op_init_args,
            op_init_kwargs=self._op_init_kwargs,
        )

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
        self._ensure_initialized()
        df = storage.read("dataframe")
        records: list[dict] = df.to_dict("records")
        run_params: dict = {"args": args, "kwargs": kwargs}
        result_records = self._module(records, run_params)
        storage.write(pd.DataFrame(result_records))

    # --- lifecycle helpers ---

    def shutdown(self) -> None:
        """Terminate all Ray actors held by this operator."""
        if self._module is None:
            return
        import ray

        for actor in self._module.actors:
            ray.kill(actor)
        self._module = None

    def __repr__(self) -> str:
        state = "initialized" if self._module is not None else "lazy"
        return (
            f"RayAcceleratedOperator({self._op_cls.__name__}, "
            f"replicas={self._replicas}, state={state})"
        )
