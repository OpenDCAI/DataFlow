from abc import ABC, abstractmethod
from dataflow.logger import get_logger

class WrapperABC(ABC):
    """
    Abstract base class for wrappers.
    """

    @abstractmethod
    def run(self) -> None:
        """
        Main function to run the wrapper.
        """
        pass