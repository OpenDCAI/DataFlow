from typing import OrderedDict, ParamSpec, TypeVar, Generic, Protocol, Any, Dict
from functools import wraps
import inspect
from dataflow.logger import get_logger
from dataflow.core import OperatorABC
# from dataflow.core.graph import nodes
from tqdm import tqdm
import pandas as pd
P = ParamSpec("P")
R = TypeVar("R")

class HasRun(Protocol[P, R]):
    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被下面动态拷贝原 operator.run 的 __doc__。
        """
        ...
        
class OPRuntime:
    def __init__(
        self, 
        operator: OperatorABC, 
        operator_name: str,
        # func: callable, 
        args: Dict[str, Any]
        ):
        self.op = operator
        self.op_name = operator_name
        # self.func = func
        self.kwargs = args
        self.logger = get_logger()

    def __repr__(self):
        # return f"OPRuntime(operator={repr(self.op)}, func={self.func.__qualname__}, args={self.kwargs})"        
        return f"OPRuntime(operator={repr(self.op)}, args={self.kwargs})"

class AutoOP(Generic[P, R]):
    """
    自动化运行 Operator 的 Wrapper。

    在静态检查/IDE 里，AutoOp.run 的签名和 operator.run 完全一致。
    运行时，会把 operator.run 的 __doc__ 和 __signature__ 也拷过来，
    这样 help(bw.run) 时能看到原 operator 的文档。
    """
    
    def __init__(self, operator: HasRun[P, R], operator_name, pipeline):
        self._operator = operator
        self._operator_name = operator_name
        self._pipeline = pipeline # <class 'dataflow.core.Pipeline.PipelineABC'>
        self._logger = get_logger()
        
        # 动态拷贝 operator.run 的 __doc__ 和 inspect.signature
        self._orig_run = operator.run
        self._signature = inspect.signature(operator.run)
        self.__doc__ = self._orig_run.__doc__

    def _flatten_bound_arguments(self, bound: inspect.BoundArguments, sig: inspect.Signature) -> "OrderedDict[str, object]":
        """
        将 bound.arguments（OrderedDict）中的 **kwargs 展平到顶层，
        保留调用顺序；其余参数（含 *args）原样保留。
        """
        # 找出签名中 **kwargs 的参数名（若存在）
        var_kw_name = None
        for name, p in sig.parameters.items():
            print(name, p.kind)
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                var_kw_name = name
                break

        out = OrderedDict()
        for name, value in bound.arguments.items():
            if name == var_kw_name and isinstance(value, dict):
                # 展平 **kwargs：按调用顺序插入
                for k, v in value.items():
                    if k in out:                   # 理论上不该冲突；保险起见
                        out[f"__kw__{k}"] = v
                    else:
                        out[k] = v
            else:
                out[name] = value
        return out

    def run(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """
        这里写一份通用的 run 文档也可以，
        不写的话会被上面动态拷贝原 operator.run 的 __doc__。
        """
        self._signature = inspect.signature(self._orig_run)
        bound_args = self._signature.bind(*args, **kwargs)
        bound_args.apply_defaults()

        final_kwargs = self._flatten_bound_arguments(bound_args, self._signature)  # OrderedDict
        (final_kwargs)
        self._logger.debug(final_kwargs)
        # final_kwargs = bound_args.arguments  # OrderedDict
        # 添加一条运行记录
        # self._pipeline.op_runtimes.append(OPRuntime(self._operator, self._orig_run, dict(final_kwargs)))
        self._pipeline.op_runtimes.append(
            OPRuntime(
                operator=self._operator, 
                operator_name=self._operator_name,
                args=dict(final_kwargs)
            )
        )