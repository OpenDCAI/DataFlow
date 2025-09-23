from .operator import OperatorABC, get_operator
from .llm_serving import LLMServingABC
from .wrapper import WrapperABC

from typing import Union, TypeAlias

# 定义类型别名
OPERATOR_CLASSES: TypeAlias = Union[OperatorABC, WrapperABC]
LLM_SERVING_CLASSES: TypeAlias = LLMServingABC  # 单一类型也可以这么写

__all__ = [
    'OPERATOR_CLASSES',
    'LLM_SERVING_CLASSES',
    'OperatorABC',
    'get_operator',
    'LLMServingABC',
    'WrapperABC',
]