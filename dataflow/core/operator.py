from abc import ABC, abstractmethod
from dataflow.logger import get_logger
from .prompt import DIYPromptABC, PromptABC

class OperatorABC(ABC):
    def __init__(self):
        self.logger = get_logger()
        self.ALLOWED_PROMPTS = tuple([type[DIYPromptABC | PromptABC]])

    @abstractmethod
    def run(self) -> None:
        """
        Main function to run the operator.
        """
        pass

def get_operator(operator_name, args) -> OperatorABC:
    from dataflow.utils import OPERATOR_REGISTRY
    print(operator_name, args)
    operator = OPERATOR_REGISTRY.get(operator_name)(args)
    logger = get_logger()
    if operator is not None:
        logger.info(f"Successfully get operator {operator_name}, args {args}")
    else:
        logger.error(f"operator {operator_name} is not found")
    assert operator is not None
    print(operator)
    return operator
