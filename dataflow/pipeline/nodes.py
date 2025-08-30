from __future__ import annotations
from dataflow.core import OperatorABC
from dataflow.core import WrapperABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OPERATOR_CLASSES, LLM_SERVING_CLASSES
from typing import Union

class KeyNode(object):
    def __init__(
        self,
        key_para_name: str,
        key: str,
        ptr: list[KeyNode] = None
    ):
        self.key_para_name = key_para_name  # name of the parameter in the operator's run functio
        self.key = key
        self.ptr = ptr if ptr != None else [] # ptr to next KeyNode(s), used to build a list of keys

    def set_index(self, index:int):
        self.index = index

    def __str__(self):
        current_id = hex(id(self))
        ptr_status = [
            (node.key, node.index, hex(id(node))) for node in self.ptr
        ] if len(self.ptr) != 0 else ["None"]
        ptr_str = "".join([
            f"\n      <{item}>" for item in ptr_status
        ])
        return f"\n    KeyNode[{current_id}](key_para_name={self.key_para_name}, key={self.key}, ptr_keys={ptr_str})"

    def __repr__(self):
        return self.__str__()

    
class OperatorNode(object):
    def __init__(
        self, 
        op_obj: OPERATOR_CLASSES = None,
        op_name: str = None, 
        storage: DataFlowStorage = None,
        llm_serving: LLM_SERVING_CLASSES = None,
        **kwargs
        ):
        self.op_obj = op_obj
        self.op_name = op_name
        self.storage = storage  # will be set when the operator is initialized
        self.llm_serving = llm_serving # will be set when the operator is initialized
        self.kwargs = kwargs  # parameters for the operator's run function
        
        # Initialize input and output keys
        self.input_keys = []
        self.input_key_nodes: dict[KeyNode] = {}
        self.output_keys = []
        self.output_keys_nodes: dict[KeyNode] = {}
        
        self._get_keys_from_kwargs()  # Extract keys from kwargs
        
    def _get_keys_from_kwargs(self):
        for k, v in self.kwargs.items():
            if k.startswith("input_") and isinstance(v, str):
                self.input_keys.append(v)
                self.input_key_nodes[v] = KeyNode(k, v)
            elif k.startswith("output_") and isinstance(v, str):
                self.output_keys.append(v)
                self.output_keys_nodes[v] = KeyNode(k, v)
            else: # warning for unexpected keys with red color
                print(f"\033[91mWarning: Unexpected key '{k}' in operator {self.op_obj.__class__.__name__}\033[0m")
        
    def init_output_keys_nodes(self, keys:list[str]):
        for key in keys:
            self.output_keys.append(key)
            self.output_keys_nodes[key]= KeyNode(key,key)
    def init_input_keys_nodes(self, keys:list[str]):
        for key in keys:
            self.input_keys.append(key)
            self.input_key_nodes[key] = KeyNode(key, key)

    def __str__(self):
        op_class = self.op_obj.__class__.__name__ if self.op_obj else "None"
        input_keys_str = ', '.join(self.input_keys)
        output_keys_str = ', '.join(self.output_keys)
        return (
            f"OperatorNode(\n"
            f"  Operator_class: {op_class},\n"
            f"  Operator_name: {self.op_name},\n"
            f"  Input Keys: [{input_keys_str}],\n"
            f"  Output Keys: [{output_keys_str}],\n"
            f"  Input Nodes: [{self.input_key_nodes}],\n"
            f"  Output Nodes: [{self.output_keys_nodes}],\n"
            f"  Additional Params: {self.kwargs}\n"
            f")"
        )
