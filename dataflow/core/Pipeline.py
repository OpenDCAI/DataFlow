from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from .Operator import OperatorABC
from .LLMServing import LLMServingABC
from dataflow.wrapper.auto_op import AutoOP
from dataflow.logger import get_logger

class PipelineABC(ABC):
    
    def __init__(self):
        self.op_runtimes = []
        self.logger = get_logger()
        self.active_llm_serving = None
        self.resource_map = defaultdict(dict)
        self.ref_map = Counter()
    
    @abstractmethod
    def forward(self):
        """
        Main Function to run the pipeline
        """
        pass
    
    def compile(self):
        # TODO construct graph of keys and count objects
        for k, v in vars(self).items():
            if isinstance(v, OperatorABC):
                setattr(self, k, AutoOP(v, self))
        self.forward()
        self.forward = self._compiled_forward
        self._build_ref_map()
        
    def _build_ref_map(self):
        for op_runtime in self.op_runtimes:
            for _, v in vars(op_runtime.op).items():
                if isinstance(v, LLMServingABC):
                    self.resource_map[op_runtime.op]["LLMServingABC"] = v
                    self.ref_map[v] += 1
                    
    def _compiled_forward(self):
        # TODO add logic for Garbage Collection of Servings
        for op_runtime in self.op_runtimes:
            is_serving_used = op_runtime.op in self.resource_map and "LLMServingABC" in self.resource_map[op_runtime.op]
            self.logger.debug(f"Ready to run {op_runtime}, is_serving_used={is_serving_used}, active_llm_serving={self.active_llm_serving}")
            if is_serving_used:
                if self.active_llm_serving:
                    self.logger.debug(f"Detected active LLM Serving {self.active_llm_serving}, cleaning up...")
                    self.active_llm_serving.cleanup()
                self.active_llm_serving = self.resource_map[op_runtime.op]["LLMServingABC"]
            op_runtime.func(**op_runtime.kwargs)
            if is_serving_used:
                self.ref_map[self.active_llm_serving] -= 1
                if self.ref_map[self.active_llm_serving] == 0:
                    self.logger.debug(f"Detected LLM Serving {self.active_llm_serving} ref reduced to 0, cleaning up...")
                    self.active_llm_serving.cleanup()
                    self.active_llm_serving = None
            
