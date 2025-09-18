import os
import inspect
import json
import pandas as pd
import gradio as gr
import importlib.util
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from dataflow.logger import get_logger
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.core import LLMServingABC
from dataflow.utils.storage import FileStorage
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================
# 1. 强制初始化注册表
# =========================
try:
    OPERATOR_REGISTRY._get_all()
except Exception as e:
    print(f"[Warning] Failed to initialize operator registry: {e}")

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

@dataclass
class PipelineInfo:
    name: str
    category: str  # 'api'
    class_obj: type
    file_path: str
    description: str = ""

class DynamicOperatorSystem:
    def __init__(self):
        self.logger = get_logger()
        self.operators: Dict[str, OperatorInfo] = {}
        self.pipelines: Dict[str, PipelineInfo] = {}
        self._discover_operators()
        self._discover_pipelines()

    def _discover_operators(self):
        try:
            for name, obj in OPERATOR_REGISTRY.get_obj_map().items():
                try:
                    info = self._analyze_operator(name, obj)
                    self.operators[name] = info
                except Exception as e:
                    print(f"[Warning] analyze operator `{name}` failed: {e}")
        except Exception as e:
            print(f"[Warning] Failed to discover operators: {e}")

    def _discover_pipelines(self):
        """只发现 API Pipelines"""
        try:
            pipeline_base = os.path.join(os.path.dirname(__file__), "..", "statics", "pipelines")
            for category in ["api_pipelines"]:
                path = os.path.join(pipeline_base, category)
                if not os.path.exists(path):
                    continue
                for fn in os.listdir(path):
                    if fn.endswith(".py") and not fn.startswith("__"):
                        try:
                            info = self._analyze_pipeline(category, fn, path)
                            if info:
                                self.pipelines[info.name] = info
                        except Exception as e:
                            print(f"[Warning] analyze pipeline `{fn}` failed: {e}")
        except Exception as e:
            print(f"[Warning] Failed to discover pipelines: {e}")

    def _extract_class_names(self, file_path):
        import ast
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if len(classes) != 1:
            self.logger.error(f"Expected exactly 1 class in {file_path}, found {classes}")
            raise ImportError(f"Multiple or no classes in {file_path}: {classes}")
        return classes[0]

    def _analyze_pipeline(self, category: str, file_name: str, dir_path: str) -> Optional[PipelineInfo]:
        file_path = os.path.join(dir_path, file_name)
        module_name = self._extract_class_names(file_path)
        try:
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                return None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls = None
            for attr in dir(module):
                obj = getattr(module, attr)
                if inspect.isclass(obj) and attr.endswith("Pipeline"):
                    cls = obj
                    break
            if not cls:
                return None
            desc = cls.__doc__ or f"{module_name} pipeline"
            return PipelineInfo(name=module_name, category="api", class_obj=cls, file_path=file_path, description=desc)
        except Exception as e:
            print(f"[Warning] Failed to analyze pipeline {file_name}: {e}")
            return None

    def _analyze_operator(self, name: str, obj: type) -> OperatorInfo:
        def extract(sig: inspect.Signature, exclude: List[str]) -> List[ParameterInfo]:
            out = []
            for pname, p in sig.parameters.items():
                if pname in exclude:
                    continue
                has_def = (p.default is not inspect._empty)
                anno = p.annotation if p.annotation is not inspect._empty else Any
                anno_str = getattr(anno, "__name__", str(anno))
                default = p.default if has_def else None
                out.append(ParameterInfo(pname, anno_str, default, has_def))
            return out

        init_sig = inspect.signature(obj.__init__)
        init_params = extract(init_sig, exclude=["self"])
        run_params = []
        if hasattr(obj, "run"):
            run_sig = inspect.signature(obj.run)
            run_params = extract(run_sig, exclude=["self", "storage"])
        desc = obj.__doc__ or ""
        return OperatorInfo(name, obj, init_params, run_params, desc)

    def get_operator_names(self) -> List[str]:
        return sorted(self.operators.keys())

    def get_pipeline_names_by_category(self, category: str) -> List[str]:
        return sorted([n for n, info in self.pipelines.items() if info.category == category])

    def get_operator_info(self, name: str) -> Optional[OperatorInfo]:
        return self.operators.get(name)

    def get_pipeline_info(self, name: str) -> Optional[PipelineInfo]:
        return self.pipelines.get(name)

    def create_serving_instance(self, serving_type: str, **kwargs) -> LLMServingABC:
        if serving_type == "API":
            api_key = kwargs.pop("api_key", None)
            api_url = kwargs.get("api_url", "https://api.openai.com/v1/chat/completions")
            if api_key:
                os.environ["DF_API_KEY"] = api_key
            from dataflow.serving import APILLMServing_request
            return APILLMServing_request(
                api_url=api_url,
                model_name=kwargs.get("model_name"),
                max_workers=int(kwargs.get("max_workers", 2)),
            )
        from dataflow.serving import LocalModelLLMServing_vllm
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
            cache_path=kwargs.get("cache_path", "./cache"),
            file_name_prefix=kwargs.get("file_name_prefix", "result"),
            cache_type=kwargs.get("cache_type", "jsonl"),
        )

dynamic_system = DynamicOperatorSystem()

# =========================
# 中英文文案字典
# =========================
zh_t = {
    "header": "# DataFlow 动态算子和Pipeline可视化",
    "operators_mode": "算子模式",
    "pipelines_mode": "Pipeline模式",
    "mode_switch": "模式切换",
    "select_operator": "选择算子",
    "select_pipeline": "选择Pipeline",
    "desc_placeholder": "描述在这里显示",
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
    "storage_section": "## 存储参数",
    "input_file_path": "输入文件路径",
    "cache_path": "缓存路径",
    "file_name_prefix": "文件名前缀",
    "run_section": "## 运行参数",
    "run_params": "运行参数 (JSON)",
    "run_button": "执行算子",
    "run_pipeline_button": "运行Pipeline",
    "summary": "执行汇总",
    "preview": "前 5 条预览（自动换行）",
    "api_pipeline_section": "### API Pipeline 配置",
    "pipeline_input_file": "Pipeline 输入文件路径",
}

en_t = {
    "header": "# DataFlow Dynamic Operators and Pipeline Visualization",
    "operators_mode": "Operators Mode",
    "pipelines_mode": "Pipeline Mode",
    "mode_switch": "Switch Mode",
    "select_operator": "Select Operator",
    "select_pipeline": "Select Pipeline",
    "desc_placeholder": "Description appears here",
    "init_section": "## Initialization Parameters (except llm_serving)",
    "other_init": "Other init params (JSON)",
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
    "run_params": "Run Params (JSON)",
    "run_button": "Run Operator",
    "run_pipeline_button": "Run Pipeline",
    "summary": "Execution Summary",
    "preview": "Top 5 Preview (wrapped)",
    "api_pipeline_section": "### API Pipeline Config",
    "pipeline_input_file": "Pipeline Input File",
}

# 默认中文
current_t = zh_t

# =========================
# 回调：切换语言
# =========================
def switch_language(lang):
    global current_t
    current_t = zh_t if lang == "中文" else en_t
    return (
        # 1. 顶部标题
        gr.update(value=current_t["header"]),
        # 2. 模式切换控件
        gr.update(choices=[current_t["operators_mode"], current_t["pipelines_mode"]],
                  label=current_t["mode_switch"],
                  value=current_t["operators_mode"]),
        # 3. 算子选择下拉
        gr.update(label=current_t["select_operator"]),
        # 4. Pipeline 选择下拉
        gr.update(label=current_t["select_pipeline"]),
        # 5. 算子描述 Markdown
        gr.update(value=current_t["desc_placeholder"]),
        # 6. Pipeline 描述 Markdown
        gr.update(value=current_t["desc_placeholder"]),
        # 7. 初始化参数区域标题
        gr.update(value=current_t["init_section"]),
        # 8. init_editor 标签
        gr.update(label=current_t["other_init"]),
        # 9. 调用方式单选
        gr.update(label=current_t["call_method"],
                  choices=[current_t["api_section"], current_t["local_section"]],
                  value=current_t["api_section"]),
        # 10. API 参数区域标题
        gr.update(value=current_t["api_section"]),
        # 11–14. API 参数输入框
        gr.update(label=current_t["api_url"]),
        gr.update(label=current_t["model_name"]),
        gr.update(label=current_t["api_key"]),
        gr.update(label=current_t["max_workers"]),
        # 15. 本地模型参数区域标题
        gr.update(value=current_t["local_section"]),
        # 16–20. 本地模型参数输入框
        gr.update(label=current_t["local_model_path"]),
        gr.update(label=current_t["tensor_parallel_size"]),
        gr.update(label=current_t["temperature"]),
        gr.update(label=current_t["top_p"]),
        gr.update(label=current_t["max_tokens"]),
        # 21. 算子存储区域标题
        gr.update(value=current_t["storage_section"]),
        # 22–24. 算子存储输入框
        gr.update(label=current_t["input_file_path"]),
        gr.update(label=current_t["cache_path"]),
        gr.update(label=current_t["file_name_prefix"]),
        # 25. 运行参数区域标题
        gr.update(value=current_t["run_section"]),
        # 26. run_editor 标签
        gr.update(label=current_t["run_params"]),
        # 27. API Pipeline 区域标题
        gr.update(value=current_t["api_pipeline_section"]),
        # 28. pipeline_input_file 标签
        gr.update(label=current_t["pipeline_input_file"]),
        # 29. Pipeline 存储区域标题
        gr.update(value=current_t["storage_section"]),
        # 30–31. Pipeline 存储输入框
        gr.update(label=current_t["cache_path"]),
        gr.update(label=current_t["file_name_prefix"]),
        # 32–35. Pipeline API 输入框
        gr.update(label=current_t["api_url"]),
        gr.update(label=current_t["model_name"]),
        gr.update(label=current_t["api_key"]),
        gr.update(label=current_t["max_workers"]),
        # 36–37. 按钮文字
        gr.update(value=current_t["run_button"]),
        gr.update(value=current_t["run_pipeline_button"]),
    )

# =========================
# 其他已有回调与函数（保持不变）
# =========================
def switch_mode(m):
    if m == current_t["operators_mode"]:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=True),
    )

def build_param_templates(op_name):
    blank = "{}"
    if not op_name:
        return (
            gr.update(value=current_t["desc_placeholder"]),
            gr.update(value=blank),
            gr.update(value=blank),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    info = dynamic_system.get_operator_info(op_name)
    if not info:
        return (
            gr.update(value="未找到算子信息" if current_t is zh_t else "Operator info not found"),
            gr.update(value=blank),
            gr.update(value=blank),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    desc = info.description or ""
    init_dict = {p.name: (p.default_value if p.has_default else "")
                 for p in info.init_params if p.name != "llm_serving"}
    run_dict  = {p.name: (p.default_value if p.has_default else "")
                 for p in info.run_params}
    init_str = json.dumps(init_dict, ensure_ascii=False, indent=2)
    run_str  = json.dumps(run_dict, ensure_ascii=False, indent=2)
    has_llm = any(p.name == "llm_serving" for p in info.init_params)
    return (
        gr.update(value=desc),
        gr.update(value=init_str),
        gr.update(value=run_str),
        gr.update(visible=has_llm),
        gr.update(visible=has_llm),
    )

def build_pipeline_info(name):
    if not name:
        return gr.update(value=current_t["desc_placeholder"])
    info = dynamic_system.get_pipeline_info(name)
    if not info:
        msg = "Pipeline 未找到" if current_t is zh_t else "Pipeline not found"
        return gr.update(value=msg)
    return gr.update(value=info.description or name)

def gradio_run(
    op_name, init_str, run_str,
    api_url, model_name, api_key, max_workers,
    local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
    input_file_path, cache_path, file_name_prefix
):
    try:
        if api_key:
            os.environ["DF_API_KEY"] = api_key
        other_init = json.loads(init_str)
        run_params = json.loads(run_str)
        info = dynamic_system.get_operator_info(op_name)
        if any(p.name == "llm_serving" for p in info.init_params):
            cfg = {"serving_type": "API", "api_url": api_url, "model_name": model_name,
                   "api_key": api_key, "max_workers": max_workers}
            other_init["llm_serving"] = dynamic_system.create_serving_instance(**cfg)
        op_inst = info.class_obj(**other_init)
        storage = dynamic_system.create_storage_instance(
            input_file_path=input_file_path,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix
        )
        op_inst.run(storage=storage.step(), **run_params)
        out_fn = storage._get_cache_file_path(storage.operator_step + 1)
        df = pd.read_json(out_fn, lines=True)
        summary = f"**{current_t['summary']}**  {len(df)} ； `{out_fn}`"
        preview = df.head(5).to_html(index=False, escape=True)
        wrapped = "<style>table{table-layout:fixed;width:100%;}td,th{white-space:pre-wrap;}</style>" + preview
        return summary, wrapped
    except Exception as e:
        return f"[Error] {e}", "<pre></pre>"

def gradio_run_pipeline(
    pipeline_name,
    pipeline_input_file,
    cache_path, file_name_prefix,
    api_url, api_model_name, api_key, api_max_workers
):
    try:
        info = dynamic_system.get_pipeline_info(pipeline_name)
        if not info:
            return f"[Error] Pipeline '{pipeline_name}' not found", "<pre></pre>"
        pipeline_inst = info.class_obj()

        pipeline_inst.storage = FileStorage(
            first_entry_file_name=pipeline_input_file or pipeline_inst.storage.first_entry_file_name,
            cache_path=cache_path,
            file_name_prefix=file_name_prefix,
            cache_type="jsonl"
        )

        if api_key:
            os.environ["DF_API_KEY"] = api_key

        new_serving = dynamic_system.create_serving_instance(
            "API",
            api_url=api_url,
            model_name=api_model_name,
            api_key=api_key,
            max_workers=api_max_workers
        )
        for comp in vars(pipeline_inst).values():
            if hasattr(comp, "llm_serving"):
                comp.llm_serving = new_serving

        pipeline_inst.forward()

        out_fn = pipeline_inst.storage._get_cache_file_path(pipeline_inst.storage.operator_step)
        if os.path.exists(out_fn):
            try:
                df = pd.read_json(out_fn, lines=True)
                summary = f"**Pipeline执行完成**  {len(df)} 条记录； {out_fn}"
                preview = df.head(5).to_html(index=False, escape=True)
                wrapped = "<style>table{table-layout:fixed;width:100%;}td,th{white-space:pre-wrap;}</style>" + preview
                return summary, wrapped
            except:
                return f"**Pipeline执行完成** 输出文件: {out_fn}", "<pre>无法预览结果</pre>"
        else:
            return "**Pipeline执行完成** 但未找到输出文件", "<pre></pre>"
    except Exception as e:
        return f"[Error] {e}", "<pre></pre>"

# =========================
# 新增：保存 Pipeline API Key 到环境变量
# =========================
def save_pipeline_api_key(key):
    import os, logging
    os.environ["DF_API_KEY"] = key
    logging.info("DF_API_KEY 已保存到环境变量")  # 终端会看到 INFO 提示


# =========================
# Gradio 界面
# =========================
with gr.Blocks(title="DataFlow 动态算子和Pipeline可视化") as demo:
    # 0. 语言切换
    lang_switch = gr.Radio(
        choices=["中文", "English"],
        value="中文",
        label="Language / 语言"
    )

    # 1. 顶部标题
    header_md = gr.Markdown(current_t["header"])

    # 2. 模式切换
    mode_switch = gr.Radio(
        choices=[current_t["operators_mode"], current_t["pipelines_mode"]],
        value=current_t["operators_mode"],
        label=current_t["mode_switch"]
    )

    # 算子模式区
    with gr.Column(visible=True) as operator_section:
        operator_dropdown = gr.Dropdown(
            choices=dynamic_system.get_operator_names(),
            label=current_t["select_operator"]
        )
        operator_desc_md = gr.Markdown(current_t["desc_placeholder"])

    # Pipeline 模式区
    with gr.Column(visible=False) as pipeline_section:
        pipeline_dropdown = gr.Dropdown(
            choices=dynamic_system.get_pipeline_names_by_category("api"),
            label=current_t["select_pipeline"]
        )
        pipeline_desc_md = gr.Markdown(current_t["desc_placeholder"])

    # 参数与存储区
    with gr.Row():
        # 算子运行区
        with gr.Column(scale=1):
            with gr.Column(visible=True) as operator_run_section:
                init_section_md = gr.Markdown(current_t["init_section"])
                init_editor = gr.Code(
                    label=current_t["other_init"],
                    language="json",
                    interactive=True,
                    lines=6,
                    value="{}"
                )
                call_method = gr.Radio(
                    choices=[current_t["api_section"], current_t["local_section"]],
                    value=current_t["api_section"],
                    label=current_t["call_method"]
                )
                with gr.Column(visible=True) as api_group:
                    api_section_md = gr.Markdown(current_t["api_section"])
                    api_url = gr.Textbox(label=current_t["api_url"])
                    model_name = gr.Textbox(label=current_t["model_name"])
                    api_key = gr.Textbox(label=current_t["api_key"], type="password")
                    max_workers = gr.Number(label=current_t["max_workers"], value=2)
                with gr.Column(visible=False) as local_group:
                    local_section_md = gr.Markdown(current_t["local_section"])
                    local_model_path = gr.Textbox(label=current_t["local_model_path"])
                    tensor_parallel_size = gr.Number(label=current_t["tensor_parallel_size"], value=1)
                    temperature = gr.Number(label=current_t["temperature"], value=0.7)
                    top_p = gr.Number(label=current_t["top_p"], value=0.9)
                    max_tokens = gr.Number(label=current_t["max_tokens"], value=1024)

        # Pipeline 运行区
        with gr.Column(visible=False) as pipeline_run_section:
            api_pipeline_section_md = gr.Markdown(current_t["api_pipeline_section"])
            pipeline_input_file = gr.Textbox(label=current_t["pipeline_input_file"], value="")
            storage_section_md2 = gr.Markdown(current_t["storage_section"])
            pipeline_cache_path = gr.Textbox(label=current_t["cache_path"], value="./cache")
            pipeline_file_name_prefix = gr.Textbox(label=current_t["file_name_prefix"], value="result")
            pipeline_api_url = gr.Textbox(label=current_t["api_url"], value="https://api.openai.com/v1/chat/completions")
            pipeline_api_model_name = gr.Textbox(label=current_t["model_name"], value="gpt-4o-mini")
            pipeline_api_key = gr.Textbox(label=current_t["api_key"], type="password")
            # 新增保存 API Key 按钮
            save_api_key_btn = gr.Button(value="保存 API Key", variant="secondary")
            # 绑定：点击按钮时执行保存环境变量的操作
            save_api_key_btn.click(
                fn=save_pipeline_api_key,
                inputs=[pipeline_api_key],
                outputs=[]
            )
            pipeline_api_max_workers = gr.Number(label=current_t["max_workers"], value=2)

        # 算子存储区
        with gr.Column(scale=1) as operator_storage_section:
            storage_section_md1 = gr.Markdown(current_t["storage_section"])
            input_file_path = gr.Textbox(label=current_t["input_file_path"], value="")
            cache_path = gr.Textbox(label=current_t["cache_path"], value="./cache")
            file_name_prefix = gr.Textbox(label=current_t["file_name_prefix"], value="result")
            run_section_md = gr.Markdown(current_t["run_section"])
            run_editor = gr.Code(
                label=current_t["run_params"],
                language="json",
                interactive=True,
                lines=4,
                value="{}"
            )

    # 按钮
    with gr.Row():
        run_btn = gr.Button(value=current_t["run_button"], variant="primary", visible=True)
        run_pipeline_btn = gr.Button(value=current_t["run_pipeline_button"], variant="primary", visible=False)

    # 输出区
    summary_md   = gr.Markdown()
    preview_html = gr.HTML()

    # 事件绑定
    lang_switch.change(
        fn=switch_language,
        inputs=[lang_switch],
        outputs=[
            header_md, mode_switch,
            operator_dropdown, pipeline_dropdown,
            operator_desc_md, pipeline_desc_md,
            init_section_md, init_editor, call_method,
            api_section_md, api_url, model_name, api_key, max_workers,
            local_section_md, local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
            storage_section_md1, input_file_path, cache_path, file_name_prefix,
            run_section_md, run_editor,
            api_pipeline_section_md, pipeline_input_file,
            storage_section_md2, pipeline_cache_path, pipeline_file_name_prefix,
            pipeline_api_url, pipeline_api_model_name, pipeline_api_key, pipeline_api_max_workers,
            run_btn, run_pipeline_btn
        ]
    )

    mode_switch.change(
        fn=switch_mode,
        inputs=[mode_switch],
        outputs=[operator_section, pipeline_section, operator_run_section, pipeline_run_section]
    ).then(
        fn=lambda m: (
            gr.update(visible=(m == current_t["operators_mode"])),
            gr.update(visible=(m != current_t["operators_mode"])),
            gr.update(visible=(m == current_t["operators_mode"]))
        ),
        inputs=[mode_switch],
        outputs=[run_btn, run_pipeline_btn, operator_storage_section]
    )

    operator_dropdown.change(
        fn=build_param_templates,
        inputs=[operator_dropdown],
        outputs=[operator_desc_md, init_editor, run_editor, api_group, local_group]
    )

    pipeline_dropdown.change(
        fn=build_pipeline_info,
        inputs=[pipeline_dropdown],
        outputs=[pipeline_desc_md]
    )

    call_method.change(
        fn=lambda m: (
            gr.update(visible=(m==current_t["api_section"])),
            gr.update(visible=(m!=current_t["api_section"]))
        ),
        inputs=[call_method],
        outputs=[api_group, local_group]
    )

    run_btn.click(
        fn=gradio_run,
        inputs=[
            operator_dropdown, init_editor, run_editor,
            api_url, model_name, api_key, max_workers,
            local_model_path, tensor_parallel_size, temperature, top_p, max_tokens,
            input_file_path, cache_path, file_name_prefix
        ],
        outputs=[summary_md, preview_html]
    )

    run_pipeline_btn.click(
        fn=gradio_run_pipeline,
        inputs=[
            pipeline_dropdown, pipeline_input_file,
            pipeline_cache_path, pipeline_file_name_prefix,
            pipeline_api_url, pipeline_api_model_name, pipeline_api_key, pipeline_api_max_workers
        ],
        outputs=[summary_md, preview_html]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, show_error=True)
