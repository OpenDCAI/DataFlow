from dataflow.utils.registry import OPERATOR_REGISTRY
from inspect import signature
from pprint import pprint
import pytest
@pytest.mark.cpu
def test_all_operator_registry():
    """
    Test function to check the operator registry.
    This will print all registered operators and their signatures.
    """
    # Get the operator map
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()
    print(OPERATOR_REGISTRY)
    
    # Check if the map is not empty
    assert len(dataflow_obj_map) > 0, "No operators found in the registry."

    # Print each operator's name and class
    for name, obj in dataflow_obj_map.items():
        print(f"Operator Name: {name}, Class: {obj.__name__}")
        if hasattr(obj, 'run'):
            run_signature = signature(obj.run)
            print(f"  run signature: {run_signature}")
        if hasattr(obj, '__init__'):
            init_signature = signature(obj.__init__)
            print(f"  __init__ signature: {init_signature}")

if __name__ == "__main__":
    # 全局table，看所有注册的算子的str名称和对应的module路径
    # 获得所有算子的类名2class映射
    OPERATOR_REGISTRY._get_all()
    print(OPERATOR_REGISTRY)
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()

    print(dataflow_obj_map)

    # 遍历所有算子，打印其名称和对象，以及init函数和run函数的签名，以及形参列表
    for name, obj in dataflow_obj_map.items():
        # use Blue color for the name
        print(f"\033[94mName: {name}, Object {obj}\033[0m")
        # get signature of the run and __init__ methods for each operator
        if hasattr(obj, 'run'):
            run_signature = signature(obj.run)
            run_signature_params = run_signature.parameters
            # green color for run method
            print("\033[92m  run signature: \033[0m")
            pprint(run_signature)
            print("\033[92m  run signature parameters: \033[0m")
            pprint(run_signature_params)
        if hasattr(obj, '__init__'):
            init_signature = signature(obj.__init__)
            init_signature_params = init_signature.parameters
            # green color for __init__ method
            print("\033[92m  __init__ signature: \033[0m")
            pprint(init_signature)
            print("\033[92m  __init__ signature parameters: \033[0m")
            pprint(init_signature_params)
        print()