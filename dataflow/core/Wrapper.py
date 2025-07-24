from abc import ABC, abstractmethod
from dataflow.core import OperatorABC
from dataflow.logger import get_logger

class WrapperABC(ABC):
    """
    Abstract base class for wrappers.
    """
    def __init__(self):
        self._operator : OperatorABC = None
    @abstractmethod
    def run(self) -> None:
        """
        Main function to run the wrapper.
        """
        pass