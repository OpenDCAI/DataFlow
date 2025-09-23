from typing import TypeVar, Protocol, Union, get_type_hints,cast
from functools import wraps
# from dataflow.core import OperatorABC

class PromptABC():
    def __init__(self):
        pass
    def build_prompt(self):
        raise NotImplementedError

class DIYPromptABC(PromptABC):
    def __init__(self):
        super().__init__()
    def build_prompt(self):
        raise NotImplementedError
    
# class OperatorWithAllowedPrompts(Protocol):
#     ALLOWED_PROMPTS: list[type[DIYPromptABC | PromptABC]]

def _make_diyprompt_union(allowed_prompts: tuple[type[PromptABC], ...]):
    """构造一个 Union 类型，包含允许的 prompt + DIYPromptABC 子类 + None"""
    return Union[tuple(allowed_prompts) + (DIYPromptABC, type(None))]

# 泛型参数，表示任意传入的 class 类型
T = TypeVar("T")

def prompt_restrict(*allowed_prompts: type[DIYPromptABC]):
    """
    装饰器：限制 prompt_template 只能是指定 Prompt 类 或 DIYPromptABC 子类
    并在运行时检查 & 更新 __annotations__（供 get_type_hints 等工具使用）
    """
    def decorator(cls:T) -> T:
        setattr(cls, "ALLOWED_PROMPTS", tuple(allowed_prompts))
        # self.ALLOWED_PROMPTS = list(allowed_prompts)

        print(allowed_prompts,"fuck!!!!")
        orig_init = cls.__init__

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            pt = kwargs.get("prompt_template", None)
            if pt is None and len(args) > 1:
                pt = args[1]

            if pt is not None and not isinstance(pt, cls.ALLOWED_PROMPTS):
                if not isinstance(pt, DIYPromptABC):
                    # 每个类的完整 import 路径，换行分隔
                    allowed_names = "\n".join(
                        f"  - {c.__module__}.{c.__qualname__}"
                        for c in cls.ALLOWED_PROMPTS
                    )
                    raise TypeError(
                        f"[{cls.__name__}] Invalid prompt_template type: {type(pt).__module__}.{type(pt).__qualname__}\n"
                        f"Expected one of:\n{allowed_names}\n"
                        f"or a custom subclass of `dataflow.core.prompt.DIYPromptABC.`"
                    )

            return orig_init(self, *args, **kwargs)

        cls.__init__ = new_init

        # 更新类型注解（运行时可见，get_type_hints 可解析）
        cls.__annotations__ = dict(getattr(cls, "__annotations__", {}))
        cls.__annotations__["prompt_template"] = _make_diyprompt_union(allowed_prompts)

        # return cast(T, cast(OperatorWithAllowedPrompts, cls))
        return cls
    return decorator