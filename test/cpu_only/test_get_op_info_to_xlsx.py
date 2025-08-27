import pandas as pd
import os
from inspect import isclass
from dataflow.utils.registry import OPERATOR_REGISTRY

def extract_primary_category(op_class):
    """
    从类的 __module__ 中提取一级分类
    例如: dataflow.operators.reasoning.generate.xxx
          -> 一级分类 = reasoning
    """
    module_path = getattr(op_class, "__module__", "")
    parts = module_path.split(".")
    if "operators" in parts:
        idx = parts.index("operators")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return "Unknown"

def build_secondary_type_map():
    """
    从 get_type_of_operator() 构建 operator_name -> secondary_type 映射
    """
    type_dict = OPERATOR_REGISTRY.get_type_of_operator()
    op_to_type = {}
    for sec_type, op_names in type_dict.items():
        for op in op_names:
            op_to_type[op] = sec_type
    return op_to_type

def export_operator_info(output_file="operators_info.xlsx", lang="zh"):
    OPERATOR_REGISTRY._get_all()
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()
    op_to_type = build_secondary_type_map()

    rows = []
    for op_name, op_class in dataflow_obj_map.items():
        if op_class is None or not isclass(op_class):
            continue

        # 一级分类：module path 里提取
        primary_type = extract_primary_category(op_class)

        # 二级分类：get_type_of_operator() 的结果
        secondary_type = op_to_type.get(op_name, "Unknown")

        # 描述
        if hasattr(op_class, "get_desc") and callable(op_class.get_desc):
            try:
                desc = op_class.get_desc(lang=lang)
            except Exception as e:
                desc = f"Error calling get_desc: {e}"
        else:
            desc = "N/A"

        rows.append({
            "一级分类": primary_type,
            "二级分类": secondary_type,
            "算子名称": op_name,
            "类路径": f"{op_class.__module__}.{op_class.__name__}",
            "算子描述": desc
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    print(f"✅ 成功导出 {len(df)} 个算子信息到 {os.path.abspath(output_file)}")

if __name__ == "__main__":
    export_operator_info("operators_info3.xlsx", lang="zh")
