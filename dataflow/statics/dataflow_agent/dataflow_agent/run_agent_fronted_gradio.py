# operator_writer_ui_gradio.py  ---- Gradio 版 UI
"""
启动方法  
-------------
1. 安装依赖  
   pip install -U gradio requests sseclient-py

2. 运行  
   python operator_writer_ui_gradio.py

3. 浏览器访问  
   http://localhost:7860
"""
import json, os, requests, contextlib
from typing import Dict, Any, Generator, Tuple

import gradio as gr
from sseclient import SSEClient  # 仅用于解析后端 /stream 接口的 Server-Sent Events


# ============ 工具函数 ============
def build_payload(
    language: str,
    target: str,
    model: str,
    session_key: str,
    json_file: str,
    py_path: str,
    api_key: str,
    chat_api: str,
    execute_operator: bool,
    use_local_model: bool,
    local_model: str,
    timeout: int,
    max_debug: int,
) -> Dict[str, Any]:
    return {
        "language": language,
        "target": target,
        "model": model,
        "sessionKEY": session_key,
        "json_file": json_file,
        "py_path": py_path,
        "api_key": api_key,
        "chat_api_url": chat_api,
        "execute_the_operator": execute_operator,
        "use_local_model": use_local_model,
        "local_model_name_or_path": local_model,
        "timeout": timeout,
        "max_debug_round": max_debug,
    }


def get_latest_operator_file(py_path: str) -> Tuple[str, str]:
    """
    在 `py_path` 的上级目录中，找出名称包含基准名且以 .py 结尾的最新文件。
    返回 (文件绝对路径, 文件内容)。失败返回 ("", "")。
    """
    dir_path = os.path.dirname(py_path)
    if not dir_path:
        return "", ""

    base_name = os.path.splitext(os.path.basename(py_path))[0]

    try:
        candidates = [
            f for f in os.listdir(dir_path)
            if f.endswith(".py") and base_name in f
        ]
    except FileNotFoundError:
        return "", ""

    if not candidates:
        return "", ""

    full_paths = [os.path.join(dir_path, f) for f in candidates]
    latest_file = max(full_paths, key=os.path.getmtime)

    with contextlib.suppress(Exception):
        with open(latest_file, "r", encoding="utf-8") as fp:
            return latest_file, fp.read()

    return "", ""


def read_cache_local(cache_dir: str = "./cache_local") -> dict:
    """
    读取 cache_local 目录下全部 .json / .jsonl 文件并返回 dict。
    结构: {filename: data}
    """
    if not os.path.isdir(cache_dir):
        return {}

    cache_data = {}
    for fn in os.listdir(cache_dir):
        if not (fn.endswith(".json") or fn.endswith(".jsonl")):
            continue
        path = os.path.join(cache_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                if fn.endswith(".json"):
                    cache_data[fn] = json.load(f)
                else:  # jsonl
                    cache_data[fn] = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            cache_data[fn] = f"<读取失败: {e}>"
    return cache_data


# ------------- 普通请求 -------------
def normal_request(
    api_base: str,
    language: str,
    model: str,
    session_key: str,
    target: str,
    json_file: str,
    py_path: str,
    api_key: str,
    chat_api: str,
    execute_operator: bool,
    use_local_model: bool,
    local_model: str,
    timeout: int,
    max_debug: int,
) -> Tuple[str, dict]:
    payload = build_payload(
        language, target, model, session_key,
        json_file, py_path, api_key, chat_api,
        execute_operator, use_local_model, local_model,
        timeout, max_debug,
    )
    try:
        r = requests.post(f"{api_base}/chatagent", json=payload, timeout=timeout + 30)
        if r.ok:
            return f"✅ HTTP {r.status_code}", r.json()
        else:
            return f"❌ HTTP {r.status_code}: {r.text}", {}
    except Exception as e:
        return f"❌ 异常: {e}", {}


# ---------- 流式请求 ----------
def stream_request(
    api_base: str,
    language: str,
    model: str,
    session_key: str,
    target: str,
    json_file: str,
    py_path: str,
    api_key: str,
    chat_api: str,
    execute_operator: bool,
    use_local_model: bool,
    local_model: str,
    timeout: int,
    max_debug: int,
) -> Generator[Tuple[str, str, dict], None, None]:
    """
    Gradio generator：实时 yield (日志, 代码, cache_local 数据)。
    """
    payload = build_payload(
        language, target, model, session_key,
        json_file, py_path, api_key, chat_api,
        execute_operator, use_local_model, local_model,
        timeout, max_debug,
    )

    whole_log, code_text, cache_data = "", "", {}
    try:
        resp = requests.post(
            f"{api_base}/chatagent/stream",
            json=payload,
            stream=True,
            timeout=None,
        )
        if resp.status_code != 200:
            yield f"❌ {resp.status_code}: {resp.text}", code_text, cache_data
            return

        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data: "):
                data = json.loads(raw[6:])
                evt = data.get("event")

                if evt == "connected":
                    line = f"🔗 连接建立: {data.get('message')}"
                elif evt == "start":
                    line = f"🛠 开始任务 `{data['task']}` …"
                elif evt == "ping":
                    line = f"⏳ {data.get('message')}"
                elif evt == "finish":
                    line = (
                        f"✅ `任务：{data['task']}` 完成，用时 {data['elapsed']} 秒\n"
                        f"结果:\n{json.dumps(data['result'], ensure_ascii=False, indent=2)}"
                    )
                elif evt == "done":
                    line = "🎉 全部任务完成"

                    # 读取最新算子文件
                    fp, content = get_latest_operator_file(py_path)
                    if content:
                        code_text = f"# 文件: {fp}\n\n{content}"

                    # 如执行了算子，则读取 cache_local
                    if execute_operator:
                        cache_data = read_cache_local()
                elif evt == "error":
                    line = f"❌ 出错: {data['detail']}"
                else:
                    line = f"{data}"

                whole_log += line + "\n\n"
                yield whole_log, code_text, cache_data

        yield whole_log, code_text, cache_data

    except Exception as e:
        yield whole_log + f"\n❌ 流式请求异常: {e}", code_text, cache_data


# ============ Gradio UI ============
with gr.Blocks(title="DataFlow-Agent · 写算子 (Gradio)") as demo:
    gr.Markdown("## 🛠️ DataFlow-Agent · 写算子 (Operator Writer)")

    with gr.Row():
        api_base = gr.Textbox(label="后端地址", value="http://localhost:8000")
        language = gr.Dropdown(["zh", "en"], value="zh", label="Language")
        model    = gr.Textbox(label="LLM Model", value="deepseek-v3")

    session_key = gr.Textbox(label="sessionKEY(会话唯一标识)", value="dataflow_demo")
    target = gr.Textbox(
        label="目标（Target）", lines=4,
        value="我需要一个算子，能够对数据进行情感分析并输出积极/消极标签。"
    )

    gr.Markdown("---")

    json_file = gr.Textbox(label="待处理的JSON文件地址",
        value="")
    py_path = gr.Textbox(label="算子代码保存路径",
        value="")
    api_key = gr.Textbox(label="DF_API_KEY", type="password",
        value="")
    chat_api = gr.Textbox(label="DF_API_URL",
        value="")

    with gr.Row():
        execute_operator = gr.Checkbox(label="执行算子", value=False)
        use_local_model  = gr.Checkbox(label="使用本地模型", value=False)

    local_model = gr.Textbox(label="本地模型路径",
        value="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct")

    with gr.Row():
        timeout   = gr.Slider(60, 7200, value=3600, step=60, label="超时 (s)")
        max_debug = gr.Slider(1, 20, value=5, step=1, label="最大 Debug 轮数")

    # ---------- 普通请求 ----------
    gr.Markdown("### 📮 普通请求（直接返回最终执行结果）")
    normal_btn  = gr.Button("发送")
    norm_status = gr.Textbox(label="状态")
    norm_output = gr.JSON(label="返回结果")

    # ---------- 流式请求 ----------
    gr.Markdown("### 🚀 流式请求（可视化agent处理过程）")
    stream_btn  = gr.Button("开始流式")
    stream_box  = gr.Textbox(lines=20, label="流式输出", interactive=False)
    code_box    = gr.Code(label="生成的算子代码 (py)", language="python", lines=22)
    cache_box   = gr.JSON(label="cache_local 算子处理之后的数据内容")

    # 事件绑定
    normal_btn.click(
        fn=normal_request,
        inputs=[api_base, language, model, session_key, target,
                json_file, py_path, api_key, chat_api,
                execute_operator, use_local_model, local_model,
                timeout, max_debug],
        outputs=[norm_status, norm_output],
    )

    stream_btn.click(
        fn=stream_request,
        inputs=[api_base, language, model, session_key, target,
                json_file, py_path, api_key, chat_api,
                execute_operator, use_local_model, local_model,
                timeout, max_debug],
        outputs=[stream_box, code_box, cache_box],
    )


# ---------- 启动 ----------
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)