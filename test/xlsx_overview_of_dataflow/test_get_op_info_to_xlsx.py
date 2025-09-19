import pandas as pd
import os
from inspect import isclass
from dataflow.utils.registry import OPERATOR_REGISTRY

def export_operator_info(output_file="operators_info.xlsx", lang="zh"):
    OPERATOR_REGISTRY._get_all()
    dataflow_obj_map = OPERATOR_REGISTRY.get_obj_map()
    op_to_type = OPERATOR_REGISTRY.get_type_of_objects()

    rows = []
    for op_name, op_class in dataflow_obj_map.items():
        if op_class is None or not isclass(op_class):
            continue

        op_type_category = op_to_type.get(op_name, "Unknown/Unknown")
        primary_type = op_type_category[0]
        secondary_type = op_type_category[1] if len(op_type_category) > 1 else "Unknown"
        third_type = op_type_category[2] if len(op_type_category) > 2 else "Unknown"
        # 描述
        if hasattr(op_class, "get_desc") and callable(op_class.get_desc):
            try:
                desc = op_class.get_desc(lang=lang)
            except Exception as e:
                desc = f"Error calling get_desc: {e}"
        else:
            desc = "N/A"

        allowed_prompt_tamplates = getattr(op_class, "ALLOWED_PROMPTS", [])
        final_prompt_tamplates = []
        for pt in allowed_prompt_tamplates:
            final_prompt_tamplates.extend([pt.__module__ + "." + pt.__name__])


        rows.append({
            "Category": primary_type,
            "Subcategory 1": secondary_type,
            "Subcategory 2": third_type,
            "Name of operator": op_name,
            "Path of class": f"{op_class.__module__}.{op_class.__name__}",
            "Operator description": desc,
            "Prompt tamplate options": str(final_prompt_tamplates)
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    print(f"✅ 成功导出 {len(df)} 个算子信息到 {os.path.abspath(output_file)}")

if __name__ == "__main__":
    # get time stamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_operator_info(f"operators_info_{timestamp}.xlsx", lang="zh")
