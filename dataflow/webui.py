import os
import inspect
import json
import pandas as pd
import gradio as gr
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import LLMServingABC
from dataflow.utils.storage import FileStorage

# =========================
# 1. 强制初始化注册表
# =========================
OPERATOR_REGISTRY._get_all()

@dataclass
class ParameterInfo:
    name: str
    type_annotation: str
    default_value: Any
    has_default: bool
    description: str = ""

@dataclass
class OperatorInfo:
    name: str
    class_obj: type
    init_params: List[ParameterInfo]
    run_params: List[ParameterInfo]
    description: str = ""

class DynamicOperatorSystem:
    def __init__(self):
        self.operators: Dict[str, OperatorInfo] = {}
        self._discover_operators()

    def _discover_operators(self):
        for name, obj in OPERATOR_REGISTRY.get_obj_map().items():
            try:
                info = self._analyze_operator(name, obj)
                self.operators[name] = info
            except Exception as e:
                print(f"[Warning] analyze operator `{name}` failed: {e}")

    def _analyze_operator(self, name: str, obj: type) -> OperatorInfo:
        def extract(sig: inspect.Signature, exclude: List[str]) -> List[ParameterInfo]:
            out = []
            for p_name, p in sig.parameters.items():
                if p_name in exclude: 
                    continue
                has_def = (p.default is not inspect._empty)
                anno = p.annotation if p.annotation is not inspect._empty else Any
                anno_str = getattr(anno, "__name__", str(anno))
                default = p.default if has_def else None
                out.append(ParameterInfo(p_name, anno_str, default, has_def))
            return out

        init_sig    = inspect.signature(obj.__init__)
        init_params = extract(init_sig, exclude=["self"])
        run_params  = []
        if hasattr(obj, "run"):
            run_sig   = inspect.signature(obj.run)
            run_params= extract(run_sig, exclude=["self","storage"])

        if hasattr(obj, "get_desc"):
            try:    desc = obj.get_desc("zh")
            except:
                try: desc = obj.get_desc("en")
                except: desc = obj.__doc__ or ""
        else:
            desc = obj.__doc__ or ""
        return OperatorInfo(name, obj, init_params, run_params, desc)

    def get_operator_names(self) -> List[str]:
        return sorted(self.operators.keys())

    def get_operator_info(self, name: str) -> Optional[OperatorInfo]:
        return self.operators.get(name)

    def create_serving_instance(self, serving_type: str, **kwargs) -> LLMServingABC:
        if serving_type == "API":
            api_key = kwargs.pop("api_key", None)
            if api_key:
                os.environ["DF_API_KEY"] = api_key
            from dataflow.serving.APILLMServing_request import APILLMServing_request
            return APILLMServing_request(
                api_url=kwargs.get("api_url"),
                model_name=kwargs.get("model_name"),
                max_workers=int(kwargs.get("max_workers", 2)),
            )
        else:
            from dataflow.serving.LocalModelLLMServing import LocalModelLLMServing_vllm
            return LocalModelLLMServing_vllm(
                hf_model_name_or_path=kwargs.get("model_path"),
                vllm_tensor_parallel_size=int(kwargs.get("tensor_parallel_size", 1)),
                vllm_temperature=float(kwargs.get("temperature", 0.7)),
                vllm_top_p=float(kwargs.get("top_p", 0.9)),
                vllm_max_tokens=int(kwargs.get("max_tokens", 1024)),
            )

    def create_storage_instance(self, **kwargs) -> FileStorage:
        return FileStorage(
            first_entry_file_name=kwargs.get("input_file_path"),
            cache_path=kwargs.get("cache_path","./cache"),
            file_name_prefix=kwargs.get("file_name_prefix","result"),
            cache_type=kwargs.get("cache_type","jsonl"),
        )

dynamic_system = DynamicOperatorSystem()


# =========================
# 翻译字典
# =========================
translations = {
    "中文": {
        "header": "# DataFlow 动态算子可视化",
        "select_operator": "选择算子",
        "desc_placeholder": "算子描述在这里显示",
        "init_section": "## 初始化参数（除 llm_serving）",
        "other_init": "其他初始化参数 (JSON)",
        "call_method": "调用方式",
        "api_section": "### API 参数",
        "api_url": "API URL",
        "model_name": "模型名称",
        "api_key": "API Key",
        "max_workers": "最大并发数",
        "local_section": "### 本地模型参数",
        "local_model_path": "本地模型路径",
        "tensor_parallel_size": "张量并行大小",
        "temperature": "Temperature",
        "top_p": "Top P",
        "max_tokens": "最大Token数",
        "storage_section": "## Storage 参数",
        "input_file_path": "输入文件路径",
        "cache_path": "缓存路径",
        "file_name_prefix": "文件名前缀",
        "run_section": "## 运行参数",
        "run_params": "运行参数 (JSON)",
        "run_button": "执行算子",
        "summary": "执行汇总",
        "preview": "前 5 条预览（自动换行）",
        "api_choice": "API",
        "local_choice": "Local",
        "language": "语言",
    },
    "English": {
        "header": "# DataFlow Dynamic Operator Visualization",
        "select_operator": "Select Operator",
        "desc_placeholder": "Operator description will appear here",
        "init_section": "## Initialization Parameters (excluding llm_serving)",
        "other_init": "Other Init Parameters (JSON)",
        "call_method": "Call Method",
        "api_section": "### API Parameters",
        "api_url": "API URL",
        "model_name": "Model Name",
        "api_key": "API Key",
        "max_workers": "Max Workers",
        "local_section": "### Local Model Parameters",
        "local_model_path": "Local Model Path",
        "tensor_parallel_size": "Tensor Parallel Size",
        "temperature": "Temperature",
        "top_p": "Top P",
        "max_tokens": "Max Tokens",
        "storage_section": "## Storage Parameters",
        "input_file_path": "Input File Path",
        "cache_path": "Cache Path",
        "file_name_prefix": "File Name Prefix",
        "run_section": "## Run Parameters",
        "run_params": "Run Parameters (JSON)",
        "run_button": "Run Operator",
        "summary": "Execution Summary",
        "preview": "Top 5 Preview (word-wrap)",
        "api_choice": "API",
        "local_choice": "Local",
        "language": "Language",
    }
}


# =========================
# 2. Gradio 界面
# =========================

def build_param_templates(op_name):
    blank = "{}"
    if not op_name:
        return (
            gr.update(value=current_t["desc_placeholder"]),
            gr.update(value=blank),
            gr.update(value=blank),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    info = dynamic_system.get_operator_info(op_name)
    if not info:
        return (
            gr.update(value="未找到算子信息") if lang=="中文" else gr.update(value="Operator not found"),
            gr.update(value=blank),
            gr.update(value=blank),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    desc = info.description or ""
    other_init = {
        p.name: (p.default_value if p.has_default else "")
        for p in info.init_params if p.name != "llm_serving"
    }
    init_str = json.dumps(other_init, ensure_ascii=False, indent=2)
    run_dict = {p.name: (p.default_value if p.has_default else "") for p in info.run_params}
    run_str = json.dumps(run_dict, ensure_ascii=False, indent=2)
    has_llm = any(p.name == "llm_serving" for p in info.init_params)
    if has_llm:
        return (
            gr.update(value=desc),
            gr.update(value=init_str),
            gr.update(value=run_str),
            gr.update(visible=True, value=current_t["api_choice"]),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    else:
        return (
            gr.update(value=desc),
            gr.update(value=init_str),
            gr.update(value=run_str),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

def update_llm_groups(method):
    if method == current_t["api_choice"]:
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)

def update_language(selected):
    global lang, current_t
    lang = selected
    current_t = translations[lang]
    # 更新所有标签和 Markdown
    return (
        gr.update(value=current_t["header"]),
        gr.update(label=current_t["select_operator"]),
        gr.update(value=current_t["desc_placeholder"]),
        gr.update(value=current_t["init_section"]),
        gr.update(label=current_t["other_init"]),
        gr.update(label=current_t["call_method"], choices=[current_t["api_choice"], current_t["local_choice"]]),
        gr.update(value=current_t["api_section"]),
        gr.update(label=current_t["api_url"]),
        gr.update(label=current_t["model_name"]),
        gr.update(label=current_t["api_key"]),
        gr.update(label=current_t["max_workers"]),
        gr.update(value=current_t["local_section"]),
        gr.update(label=current_t["local_model_path"]),
        gr.update(label=current_t["tensor_parallel_size"]),
        gr.update(label=current_t["temperature"]),
        gr.update(label=current_t["top_p"]),
        gr.update(label=current_t["max_tokens"]),
        gr.update(value=current_t["storage_section"]),
        gr.update(label=current_t["input_file_path"]),
        gr.update(label=current_t["cache_path"]),
        gr.update(label=current_t["file_name_prefix"]),
        gr.update(value=current_t["run_section"]),
        gr.update(label=current_t["run_params"]),
        gr.update(value=current_t["run_button"]),
        gr.update(label=current_t["summary"]),
        gr.update(label=current_t["preview"]),
        gr.update(label=current_t["language"]),
    )

def gradio_run(
    op_name, init_str, run_str,
    method,
    api_url, model_name, api_key, max_workers,
    local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
    input_file_path, cache_path, file_name_prefix
):
    try:
        other_init = json.loads(init_str)
        run_params = json.loads(run_str)
        info = dynamic_system.get_operator_info(op_name)
        # llm_serving
        if any(p.name == "llm_serving" for p in info.init_params):
            cfg = {"type": method}
            if method == current_t["api_choice"]:
                cfg.update(api_url=api_url, model_name=model_name, api_key=api_key, max_workers=max_workers)
            else:
                cfg.update(model_path=local_model_path, tensor_parallel_size=tensor_parallel_size,
                           temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            t = cfg.pop("type")
            other_init["llm_serving"] = dynamic_system.create_serving_instance(t, **cfg)
        # storage
        run_params.update({
            "input_file_path": input_file_path,
            "cache_path": cache_path,
            "file_name_prefix": file_name_prefix,
            "cache_type": "jsonl"
        })
        # run
        op_inst = info.class_obj(**other_init)
        storage = dynamic_system.create_storage_instance(**run_params)
        op_inst.run(
            storage=storage.step(),
            **{k: v for k, v in run_params.items() if k not in {"input_file_path","cache_path","file_name_prefix","cache_type"}}
        )
        out_fn = storage._get_cache_file_path(storage.operator_step+1)
        df = pd.read_json(out_fn, lines=True)
        summary = f"**{current_t['summary']}**  {len(df)} ； `{out_fn}`"
        preview = df.head(5).to_html(index=False, escape=True)
        wrapped = (
            "<style>table{table-layout:fixed;width:100%;}td,th{white-space:pre-wrap;word-wrap:break-word;}</style>"
            + preview
        )
        return summary, wrapped
    except Exception as e:
        return f"[Error] {str(e)}", "<pre></pre>"

# 全局当前语言 & 文本
lang = "中文"
current_t = translations[lang]

with gr.Blocks(title="DataFlow 动态算子可视化") as demo:
    # 语言开关
    language = gr.Radio(
        choices=["中文","English"], value=lang, label=current_t["language"]
    )
    # 标题
    header_md = gr.Markdown(current_t["header"])

    with gr.Row():
        dropdown = gr.Dropdown(
            choices=dynamic_system.get_operator_names(),
            label=current_t["select_operator"]
        )
        desc_md = gr.Markdown(current_t["desc_placeholder"])

    with gr.Row():
        with gr.Column(scale=1):
            init_section_md = gr.Markdown(current_t["init_section"])
            init_editor     = gr.Code(label=current_t["other_init"], language="json", interactive=True, lines=8, value="{}")
            call_method     = gr.Radio(choices=[current_t["api_choice"],current_t["local_choice"]],
                                      value=current_t["api_choice"], label=current_t["call_method"], visible=False)
            with gr.Column(visible=False) as api_group:
                api_section_md = gr.Markdown(current_t["api_section"])
                api_url        = gr.Textbox(label=current_t["api_url"])
                model_name     = gr.Textbox(label=current_t["model_name"])
                api_key        = gr.Textbox(label=current_t["api_key"], type="password")
                max_workers    = gr.Number(label=current_t["max_workers"], value=2)
            with gr.Column(visible=False) as local_group:
                local_section_md    = gr.Markdown(current_t["local_section"])
                local_model_path    = gr.Textbox(label=current_t["local_model_path"])
                tensor_parallel_size= gr.Number(label=current_t["tensor_parallel_size"], value=1)
                temperature         = gr.Number(label=current_t["temperature"], value=0.7)
                top_p               = gr.Number(label=current_t["top_p"], value=0.9)
                max_tokens          = gr.Number(label=current_t["max_tokens"], value=1024)

        with gr.Column(scale=1):
            storage_section_md = gr.Markdown(current_t["storage_section"])
            input_file_path    = gr.Textbox(label=current_t["input_file_path"], value="")
            cache_path         = gr.Textbox(label=current_t["cache_path"], value="./cache")
            file_name_prefix   = gr.Textbox(label=current_t["file_name_prefix"], value="result")
            run_section_md     = gr.Markdown(current_t["run_section"])
            run_editor         = gr.Code(label=current_t["run_params"], language="json", interactive=True, lines=6, value="{}")

    run_btn     = gr.Button(current_t["run_button"], variant="primary", size="lg")
    summary_md  = gr.Markdown()
    preview_html= gr.HTML()

    # 绑定
    language.change(
        fn=update_language,
        inputs=[language],
        outputs=[
            header_md,
            dropdown, desc_md,
            init_section_md, init_editor, call_method,
            api_section_md, api_url, model_name, api_key, max_workers,
            local_section_md, local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
            storage_section_md, input_file_path, cache_path, file_name_prefix,
            run_section_md, run_editor,
            run_btn, summary_md, preview_html, language
        ]
    )
    dropdown.change(
        fn=build_param_templates,
        inputs=[dropdown],
        outputs=[desc_md, init_editor, run_editor, call_method, api_group, local_group]
    )
    call_method.change(
        fn=update_llm_groups,
        inputs=[call_method],
        outputs=[api_group, local_group]
    )
    run_btn.click(
        fn=gradio_run,
        inputs=[
            dropdown, init_editor, run_editor,
            call_method,
            api_url, model_name, api_key, max_workers,
            local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
            input_file_path, cache_path, file_name_prefix
        ],
        outputs=[summary_md, preview_html]
    )

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, show_error=True)
