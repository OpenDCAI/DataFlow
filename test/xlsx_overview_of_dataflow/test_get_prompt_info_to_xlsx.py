import pandas as pd
import os
from inspect import isclass
from inspect import isclass, getmembers, isfunction
from dataflow.utils.registry import PROMPT_REGISTRY, OPERATOR_REGISTRY

def export_prompt_info(output_file="prompt_info.xlsx", lang="zh"):
    # PROMPT_REGISTRY._get_all()
    OPERATOR_REGISTRY._get_all()
    # get mapping of prompt name to class
    prompt2operator_str = {}
    for op_name, op_class in OPERATOR_REGISTRY.get_obj_map().items():
        allowed_prompts = getattr(op_class, "ALLOWED_PROMPTS", [])
        for pt in allowed_prompts:
            prompt2operator_str[pt.__name__] = op_name


    dataflow_obj_map = PROMPT_REGISTRY.get_obj_map()
    prompt_to_type = PROMPT_REGISTRY.get_type_of_objects()

    rows = []
    for prompt_name, prompt_class in dataflow_obj_map.items():
        if prompt_class is None or not isclass(prompt_class):
            continue

        op_type_category = prompt_to_type.get(prompt_name, "Unknown/Unknown")
        primary_type = op_type_category[0]
        secondary_type = op_type_category[1] if len(op_type_category) > 1 else "Unknown"
        third_type = op_type_category[2] if len(op_type_category) > 2 else "Unknown"
        # 描述
        if hasattr(prompt_class, "get_desc") and callable(prompt_class.get_desc):
            try:
                desc = prompt_class.get_desc(lang=lang)
            except Exception as e:
                desc = f"Error calling get_desc: {e}"
        else:
            desc = "N/A"

        if prompt_name in prompt2operator_str:
            op_str_called_by_this_prompt = prompt2operator_str[prompt_name]
            op_obj_called_by_this_prompt = OPERATOR_REGISTRY.get_obj_map().get(op_str_called_by_this_prompt, None)
            op_full_path = f"{op_obj_called_by_this_prompt.__module__}.{op_obj_called_by_this_prompt.__name__}" if op_obj_called_by_this_prompt else "N/A"
        else:
            op_full_path = "N/A"



        # ✅ 获取该类所有成员函数（不包含继承自 object 的）
        member_functions = [
            name for name, func in getmembers(prompt_class, predicate=isfunction)
            if func.__qualname__.startswith(prompt_class.__name__)
        ]
        member_functions_str = ",".join(member_functions) if member_functions else "N/A"

        rows.append({
            "Category": primary_type,
            "Subcategory 1": secondary_type,
            "Subcategory 2": third_type,
            "name of prompt": prompt_name,
            "path of class": f"{prompt_class.__module__}.{prompt_class.__name__}",
            "Member functions": member_functions_str,
            "Prompt description": desc,
            "Used by operator": op_full_path
        })

    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)
    print(f"✅ 成功导出 {len(df)} 个Prompt信息到 {os.path.abspath(output_file)}")



if __name__ == "__main__":
    # get time stamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    export_prompt_info(f"prompts_info_{timestamp}.xlsx", lang="zh")
