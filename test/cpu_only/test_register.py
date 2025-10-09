from dataflow.utils.registry import OPERATOR_REGISTRY, PROMPT_REGISTRY
from inspect import signature
from pprint import pprint
import pytest
from inspect import isclass, getmembers, isfunction

def build_tree(type_dict):
    """
    根据 type_dict 构建层级统计树
    """
    tree = {}

    for op_name, categories in type_dict.items():
        node = tree
        for cat in categories:
            if cat not in node:
                node[cat] = {"__count__": 0, "__children__": {}}
            node[cat]["__count__"] += 1
            node = node[cat]["__children__"]
    return tree


def print_tree(tree, indent=0):
    """
    递归打印树状统计结果
    """
    for cat, info in tree.items():
        print("  " * indent + f"- {cat} ({info['__count__']})")
        print_tree(info["__children__"], indent + 1)


@pytest.mark.cpu
def test_all_operator_registry():
    """
    Test function to check the operator registry.
    This will print all registered operators and their signatures.
    """
    # Get the operator map
    OPERATOR_REGISTRY._get_all()
    print(OPERATOR_REGISTRY)
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()

    # pprint(dataflow_obj_map)
    # print typedict of all operators
    print("\nTypedict of all operators:")
    type_dict = OPERATOR_REGISTRY.get_type_of_objects()
    # pprint(type_dict)

    # ---- 数量匹配检查 ----
    num_by_typedict = len(type_dict)   # 新格式：key 就是 operator
    num_by_registry = len(dataflow_obj_map)

    print(f"\n{num_by_typedict} operators in total by type dict.")
    print(f"{num_by_registry} operators registered in the registry.")

    if num_by_typedict != num_by_registry:
        print("Mismatch found:")
        # 找出 type_dict 有但 registry 没有的
        for op in type_dict.keys():
            if op not in dataflow_obj_map:
                print(f"  [Missing in registry] {op}")
        # 找出 registry 有但 type_dict 没有的
        for op in dataflow_obj_map.keys():
            if op not in type_dict:
                print(f"  [Missing in type_dict] {op}")

    # ---- 层级统计树 ----
    tree = build_tree(type_dict)
    print("\nOperator Type Hierarchy Statistics:")
    print_tree(tree)

    # ---- 逐个打印信息 ----
    assert len(dataflow_obj_map) > 0, "No operators found in the registry."
    # for name, obj in dataflow_obj_map.items():
    #     print(f"\nOperator Name: {name}, Class: {obj.__name__}")
    #     if hasattr(obj, 'run'):
    #         run_signature = signature(obj.run)
    #         print(f"  run signature: {run_signature}")
    #     if hasattr(obj, '__init__'):
    #         init_signature = signature(obj.__init__)
    #         print(f"  __init__ signature: {init_signature}")
    
    # prompt registry
    print("\nPrompt Registry:")
    # PROMPT_REGISTRY._get_all() # will cause bug and ERROR
    print(PROMPT_REGISTRY)
    prompt_type_dict = PROMPT_REGISTRY.get_type_of_objects()
    print("\nPrompt Type Hierarchy Statistics:")
    print_tree(build_tree(prompt_type_dict))

    # 成员函数检测逻辑
    print("\n🔍 Checking Prompt class member functions ...")
    allowed_methods = {"build_prompt", "__init__", "build_system_prompt"}
    invalid_prompts = []

    prompt_map = PROMPT_REGISTRY.get_obj_map()
    for name, cls in prompt_map.items():
        if cls is None or not isclass(cls):
            continue

        # 获取类中定义的成员函数（排除继承）
        member_funcs = [
            func_name for func_name, func_obj in getmembers(cls, predicate=isfunction)
            if func_obj.__qualname__.startswith(cls.__name__)
        ]

        # 找出不被允许的方法
        disallowed = [
            fn for fn in member_funcs
            if not (fn in allowed_methods or fn.startswith("_"))
        ]

        if disallowed:
            invalid_prompts.append((name, cls.__module__, disallowed))

    # 报告结果
    if invalid_prompts:
        print("\n❌ Check failed, invalid Prompt classes contain disallowed functions:")
        for name, module, funcs in invalid_prompts:
            print(f"- {name} ({module}) disallowed functions: {funcs}")

        # 构造详细错误说明
        rule_explanation = (
            "\nPrompt class naming rule (English):\n"
            "Each Prompt class is only allowed to define the following public methods:\n"
            "  - build_prompt\n"
            "  - build_system_prompt\n"
            "  - __init__\n"
            "Other methods are only allowed if they start with an underscore (_), "
            "indicating they are private helper methods.\n\n"
            "Please check all invalid Prompt classes.\n"
        )

        # 详细列出问题
        details = "\n".join(
            f"  • {name} ({module}) → invalid functions: {funcs}"
            for name, module, funcs in invalid_prompts
        )

        errors = []
        errors.append(
            f"❌ Found {len(invalid_prompts)} Prompt classes violating naming rules.\n"
            f"{rule_explanation}\n"
            f"Details:\n{details}"
        )

    else:
        print("✅ All Prompt class member functions comply with the conventions (only contain allowed functions or private functions)")

    if errors:
        pytest.fail("\n".join(errors), pytrace=False)

if __name__ == "__main__":
    # 全局table，看所有注册的算子的str名称和对应的module路径
    # 获得所有算子的类名2class映射
    # Get the operator map

    test_all_operator_registry()
    exit(0)

    OPERATOR_REGISTRY._get_all()
    print(OPERATOR_REGISTRY)
    # from dataflow.operators.chemistry import ExtractSmilesFromText
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()
    print(OPERATOR_REGISTRY)
    # print count
    print("Total number of OPERATORS:",len(dataflow_obj_map))


    from dataflow.utils.registry import PROMPT_REGISTRY
    print(PROMPT_REGISTRY)


    from dataflow.operators.core_text import PromptedGenerator

    from pprint import pprint
    pprint(OPERATOR_REGISTRY.get_type_of_objects())
    # 因为多个prompt在同一个路径下，所以最后一个module的字段是总的，而非具体prompt的名字。
    pprint(PROMPT_REGISTRY.get_type_of_objects())
    
    # pprint(dataflow_obj_map)
    # # print typedict of all operators
    # print("\nTypedict of all operators:")
    # type_dict = OPERATOR_REGISTRY.get_type_of_operator()
    # pprint(type_dict)   
    # print(len(dataflow_obj_map), "operators registered in the registry.")

    # type_dict_set = set([q for k, v in type_dict.items() for q in v])

    # sum_of_types = sum(len(v) for v in type_dict.values())
    # print(sum_of_types, "operators in total by type dict.")
    # if sum_of_types != len(dataflow_obj_map):
    #     # Raise a warning if the sum of types does not match the total number of operator
    #     # raise Warning("The sum of types does not match the total number of operators.")
    #     # check which one is not matching
    #     print("Mismatch found:")
    #     for key, value in type_dict.items():
    #         for operator in value:
    #             if operator not in dataflow_obj_map:
    #                 raise Warning(f"Operator `{operator}` in type dict  not found in the registry.")
    #     for operator in dataflow_obj_map:
    #         if operator not in type_dict_set:
    #             raise Warning(f"Operator `{operator}` in registry not found in the type dict.")


    # for key, value in type_dict.items():
    #     print(f"{key}: {len(value)} operators")
    # # Check if the map is not empty
    # assert len(dataflow_obj_map) > 0, "No operators found in the registry."

    # # 遍历所有算子，打印其名称和对象，以及init函数和run函数的签名，以及形参列表
    # for name, obj in dataflow_obj_map.items():
    #     # use Blue color for the name
    #     print(f"\033[94mName: {name}, Object {obj}\033[0m")
    #     # get signature of the run and __init__ methods for each operator
    #     if hasattr(obj, 'run'):
    #         run_signature = signature(obj.run)
    #         run_signature_params = run_signature.parameters
    #         # green color for run method
    #         print("\033[92m  run signature: \033[0m")
    #         pprint(run_signature)
    #         print("\033[92m  run signature parameters: \033[0m")
    #         pprint(run_signature_params)
    #     if hasattr(obj, '__init__'):
    #         init_signature = signature(obj.__init__)
    #         init_signature_params = init_signature.parameters
    #         # green color for __init__ method
    #         print("\033[92m  __init__ signature: \033[0m")
    #         pprint(init_signature)
    #         print("\033[92m  __init__ signature parameters: \033[0m")
    #         pprint(init_signature_params)
    #     print()