#!/usr/bin/env python3
#  —— FastAPI + Gradio 同进程启动（UI = /ui, API = /api）

import os, json, contextlib, requests
from typing import Dict, Any, Generator, Tuple

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn
from run_dataflow_agent_with_ui import app as backend_app
def build_payload(
    language, target, model, session_key,
    json_file, py_path, api_key, chat_api,
    execute_operator, execute_pipeline,
    use_local_model, local_model,
    timeout, max_debug,
):
    return dict(
        language=language, target=target, model=model, sessionKEY=session_key,
        json_file=json_file, py_path=py_path, api_key=api_key,
        chat_api_url=chat_api, execute_the_operator=execute_operator,
        execute_the_pipeline=execute_pipeline,
        use_local_model=use_local_model,
        local_model_name_or_path=local_model,
        timeout=timeout, max_debug_round=max_debug,
    )

def get_latest_operator_file(py_path):
    dir_path = os.path.dirname(py_path)
    if not dir_path:
        return "", ""
    base_name = os.path.splitext(os.path.basename(py_path))[0]
    try:
        candidates = [f for f in os.listdir(dir_path) if f.endswith(".py") and base_name in f]
    except FileNotFoundError:
        return "", ""
    if not candidates:
        return "", ""
    full_paths = [os.path.join(dir_path, f) for f in candidates]
    latest = max(full_paths, key=os.path.getmtime)
    with contextlib.suppress(Exception):
        with open(latest, "r", encoding="utf-8") as f:
            return latest, f.read()
    return "", ""

def read_cache_local(cache_dir: str = "./cache_local"):
    if not os.path.isdir(cache_dir):
        return {}
    out = {}
    for fn in os.listdir(cache_dir):
        if not (fn.endswith(".json") or fn.endswith(".jsonl")):
            continue
        path = os.path.join(cache_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                out[fn] = json.load(f) if fn.endswith(".json") else [json.loads(x) for x in f if x.strip()]
        except Exception as e:
            out[fn] = f"<读取失败: {e}>"
    return out

# ------------------------------------------------------------------
# 与后端交互：普通 / 流式
# ------------------------------------------------------------------
def normal_request(
    api_base, language, model, session_key, target,
    json_file, py_path, api_key, chat_api,
    execute_operator, execute_pipeline,
    use_local_model, local_model,
    timeout, max_debug,
):
    payload = build_payload(language, target, model, session_key,
                            json_file, py_path, api_key, chat_api,
                            execute_operator, execute_pipeline,
                            use_local_model, local_model,
                            timeout, max_debug)
    try:
        r = requests.post(f"{api_base}/chatagent", json=payload, timeout=timeout + 30)
        return (f"✅ HTTP {r.status_code}", r.json()) if r.ok else (f"❌ HTTP {r.status_code}: {r.text}", {})
    except Exception as e:
        return f"❌ 异常: {e}", {}

def stream_request(
    api_base, language, model, session_key, target,
    json_file, py_path, api_key, chat_api,
    execute_operator, execute_pipeline,
    use_local_model, local_model,
    timeout, max_debug,
):
    payload = build_payload(language, target, model, session_key,
                            json_file, py_path, api_key, chat_api,
                            execute_operator, execute_pipeline,
                            use_local_model, local_model,
                            timeout, max_debug)
    whole_log, code_txt, cache = "", "", {}
    try:
        resp = requests.post(f"{api_base}/chatagent/stream", json=payload, stream=True, timeout=None)
        if resp.status_code != 200:
            yield f"❌ {resp.status_code}: {resp.text}", code_txt, cache
            return
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = json.loads(raw[6:])
            evt = data.get("event")
            if evt == "connected":
                line = f"🔗 连接建立: {data['message']}"
            elif evt == "start":
                line = f"🛠 开始任务 `{data['task']}` …"
            elif evt == "ping":
                line = f"⏳ {data.get('message')}"
            elif evt == "finish":
                line = (f"✅ 任务 `{data['task']}` 完成，耗时 {data['elapsed']} 秒\n"
                        f"{json.dumps(data['result'], ensure_ascii=False, indent=2)}")
            elif evt == "done":
                line = "🎉 全部任务完成"
                fp, content = get_latest_operator_file(py_path)
                if content:
                    code_txt = f"# 文件: {fp}\n\n{content}"
                if execute_operator or execute_pipeline:
                    cache = read_cache_local()
            elif evt == "error":
                line = f"❌ 出错: {data['detail']}"
            else:
                line = str(data)
            whole_log += line + "\n\n"
            yield whole_log, code_txt, cache
        yield whole_log, code_txt, cache
    except Exception as e:
        yield whole_log + f"\n❌ 流式异常: {e}", code_txt, cache

with gr.Blocks(title="DataFlow-Agent") as demo:
    gr.Markdown("## 🛠️ DataFlow-Agent 算子编写 + 管线推荐")

    with gr.Row():
        api_base = gr.Textbox(label="后端地址", value="http://127.0.0.1:7860/api")
        language = gr.Dropdown(["zh", "en"], value="zh", label="Language")
        model    = gr.Textbox(label="LLM Model", value="deepseek-v3")

    session_key = gr.Textbox(label="sessionKEY", value="dataflow_demo")
    target = gr.Textbox(label="目标（Target）", lines=4,
                        value="我需要一个算子，使用LLMServing对医疗场景的原始题干进行同义改写，生成语义一致但表达不同的新问题，有效增加训练样本多样性，并且输入key是question，输出key是questionPARA,就在原数据上新加入key。")

    gr.Markdown("---")

    json_file = gr.Textbox(label="待处理 JSON 文件地址")
    py_path   = gr.Textbox(label="算子代码保存路径 (.py)")
    api_key   = gr.Textbox(label="DF_API_KEY", type="password")
    chat_api  = gr.Textbox(label="DF_API_URL")

    with gr.Row():
        execute_operator = gr.Checkbox(label="调试算子（耗 tokens）")
        execute_pipeline = gr.Checkbox(label="调试 pipeline（耗 tokens）")
        use_local_model  = gr.Checkbox(label="使用本地模型")

    local_model = gr.Textbox(label="本地模型路径",
                             value="/mnt/public/model/huggingface/Qwen2.5-7B-Instruct")

    with gr.Row():
        timeout   = gr.Slider(60, 7200, value=3600, step=60, label="超时 (s)")
        max_debug = gr.Slider(1, 20, value=5, step=1, label="最大 Debug 轮数")

    gr.Markdown("### 📮 普通请求")
    normal_btn  = gr.Button("发送")
    norm_status = gr.Textbox(label="状态")
    norm_output = gr.JSON(label="返回结果")

    gr.Markdown("### 🚀 流式请求")
    stream_btn  = gr.Button("开始流式")
    stream_box  = gr.Textbox(lines=20, label="流式输出", interactive=False)
    code_box    = gr.Code(label="生成的算子代码", language="python", lines=22)
    cache_box   = gr.JSON(label="cache_local 数据")

    normal_btn.click(
        normal_request,
        inputs=[api_base, language, model, session_key, target,
                json_file, py_path, api_key, chat_api,
                execute_operator, execute_pipeline,
                use_local_model, local_model,
                timeout, max_debug],
        outputs=[norm_status, norm_output],
    )

    stream_btn.click(
        stream_request,
        inputs=[api_base, language, model, session_key, target,
                json_file, py_path, api_key, chat_api,
                execute_operator, execute_pipeline,
                use_local_model, local_model,
                timeout, max_debug],
        outputs=[stream_box, code_box, cache_box],
    )

# ------------------------------------------------------------------
# FastAPI 组合
# ------------------------------------------------------------------
root = FastAPI()

root.mount("/api", backend_app)

gr.mount_gradio_app(root, demo, path="/ui")

@root.get("/", include_in_schema=False)
async def _to_ui():
    return RedirectResponse("/ui")

@root.get("/manifest.json", include_in_schema=False)
async def _manifest():
    return JSONResponse({"name": "DataFlow-Agent", "start_url": ".", "display": "standalone"})

# ------------------------------------------------------------------
# 运行
# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(root, host="0.0.0.0", port=7860)