from abc import ABC, abstractmethod
from typing import Any, List


class LLMServingABC(ABC):
    """Abstract base class for data generators. Which may be used to generate data from a model or API. Called by operators
    """    
    @abstractmethod
    def generate_from_input(self, user_inputs: List[str], system_prompt: str) -> List[str]:
        """
        Generate data from input.
        input: List[str], the input of the generator
        """
        pass
    
    @abstractmethod
    def start_serving(self):
        """
        Cleanup the generator and garbage collect all GPU/CPU memory.
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup the generator and garbage collect all GPU/CPU memory.
        """
        pass
    
    def load_model(self, model_name_or_path: str, **kwargs: Any):
        """
        Load the model from the given path.
        This method is optional and can be overridden by subclasses if needed.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
