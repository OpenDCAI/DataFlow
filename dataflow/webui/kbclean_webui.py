import gradio as gr
import os
import shutil
import pandas as pd
from typing import List
from contextlib import redirect_stdout, redirect_stderr
import io
import threading
import time
from dataflow.webui.kbcleaning_tools import KBCleaning_Tools

CACHE_DIR = "./.cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def upload_files(files: List[str], current_list: List[str]):
    new_list = current_list.copy()
    for fp in files:
        if not fp.lower().endswith(".pdf"):
            continue
        dst = os.path.join(CACHE_DIR, os.path.basename(fp))
        shutil.copy(fp, dst)
        if dst not in new_list:
            new_list.append(dst)
    return new_list, pd.DataFrame(new_list, columns=["PDF Files"])


def add_pdf_path(path: str, current_list: List[str]):
    new_list = current_list.copy()
    if path and path.lower().endswith(".pdf") and os.path.exists(path):
        if path not in new_list:
            new_list.append(path)
    return new_list, pd.DataFrame(new_list, columns=["PDF Files"])


def add_url(url: str, current_list: List[str]):
    new_list = current_list.copy()
    if url and url not in new_list:
        new_list.append(url)
    return new_list, pd.DataFrame(new_list, columns=["URLs"])


def run_pipeline_ui(pdf_list: List[str], url_list: List[str],
                    serving_type: str,
                    model_name: str,
                    vllm_max_tokens: int, vllm_tensor_parallel: int,
                    api_url: str, api_key: str, api_max_workers: int,
                    sglang_max_tokens: int, sglang_tensor_parallel: int, sglang_max_workers: int, sglang_temperature: float):
    kb = KBCleaning_Tools()
    inputs = pdf_list + url_list
    out_buffer = io.StringIO()

    # Choose parameters based on serving type
    if serving_type == "vllm":
        max_tokens = vllm_max_tokens
        tensor_parallel_size = vllm_tensor_parallel
        max_workers = None
        temperature = None
        _api_url, _api_key = None, None
    elif serving_type == "api":
        # Save provided API key into environment variable
        os.environ["DF_API_KEY"] = api_key
        max_tokens = None
        tensor_parallel_size = None
        max_workers = api_max_workers
        temperature = None
        _api_url, _api_key = api_url, "DF_API_KEY"
    else:  # sglang
        max_tokens = sglang_max_tokens
        tensor_parallel_size = sglang_tensor_parallel
        max_workers = sglang_max_workers
        temperature = sglang_temperature
        _api_url, _api_key = None, None

    def target():
        with redirect_stdout(out_buffer), redirect_stderr(out_buffer):
            kb.run_pipeline(
                pdf_or_url_list=inputs,
                serving_type=serving_type,
                model_name=model_name,
                max_tokens=max_tokens,
                tensor_parallel_size=tensor_parallel_size,
                api_url=_api_url,
                api_key=_api_key,
                max_workers=max_workers,
                temperature=temperature
            )
    thread = threading.Thread(target=target)
    thread.start()

    while thread.is_alive():
        time.sleep(0.2)
        lines = out_buffer.getvalue().splitlines()
        yield "\n".join(lines[-10:])
    lines = out_buffer.getvalue().splitlines()
    yield "\n".join(lines[-10:])


def update_param_visibility(selected):
    return (
        gr.update(visible=(selected == "vllm")),
        gr.update(visible=(selected == "api")),
        gr.update(visible=(selected == "sglang"))
    )


def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Knowledge Cleaning Pipeline")

        # File and URL inputs
        file_input = gr.File(label="Drag & Drop PDF files", file_count="multiple", type="filepath")
        pdf_path_input = gr.Textbox(label="Or input PDF path", placeholder="e.g. /path/to/doc.pdf")
        add_pdf_btn = gr.Button("Add PDF Path")
        url_input = gr.Textbox(label="Input URL", placeholder="e.g. https://example.com/doc")
        add_url_btn = gr.Button("Add URL")
        pdf_table = gr.DataFrame(headers=["PDF Files"])
        url_table = gr.DataFrame(headers=["URLs"])
        pdf_state = gr.State([])
        url_state = gr.State([])

        # Serving type selector
        serving_type = gr.Radio(choices=["vllm", "api", "sglang"], value="vllm", label="Serving Type")

        # Common parameter
        model_name = gr.Textbox(value="Qwen/Qwen2.5-7B-Instruct", label="Model Name")

        # Parameter groups
        vllm_params = gr.Column(visible=True)
        with vllm_params:
            vllm_max_tokens = gr.Number(value=2048, label="Max Tokens")
            vllm_tensor_parallel = gr.Number(value=1, label="Tensor Parallel Size")

        api_params = gr.Column(visible=False)
        with api_params:
            api_url = gr.Textbox(label="API URL", placeholder="e.g. http://api.server")
            api_key = gr.Textbox(label="API Key", type="password")
            api_max_workers = gr.Number(value=1, label="Max Workers")

        sglang_params = gr.Column(visible=False)
        with sglang_params:
            sglang_max_tokens = gr.Number(value=2048, label="Max New Tokens (SGLang)")
            sglang_tensor_parallel = gr.Number(value=1, label="Tensor Parallel Size")
            sglang_max_workers = gr.Number(value=1, label="Max Workers")
            sglang_temperature = gr.Number(value=0.7, label="Temperature")

        # Bind visibility update
        serving_type.change(update_param_visibility,
                             inputs=[serving_type],
                             outputs=[vllm_params, api_params, sglang_params])

        # Run button and output
        run_btn = gr.Button("Run Pipeline")
        output = gr.Textbox(label="Console Output", interactive=False, lines=10)

        # Event bindings
        file_input.change(upload_files,
                          inputs=[file_input, pdf_state],
                          outputs=[pdf_state, pdf_table])
        add_pdf_btn.click(add_pdf_path,
                          inputs=[pdf_path_input, pdf_state],
                          outputs=[pdf_state, pdf_table])
        add_url_btn.click(add_url,
                         inputs=[url_input, url_state],
                         outputs=[url_state, url_table])
        run_btn.click(run_pipeline_ui,
                      inputs=[pdf_state, url_state,
                              serving_type, model_name,
                              vllm_max_tokens, vllm_tensor_parallel,
                              api_url, api_key, api_max_workers,
                              sglang_max_tokens, sglang_tensor_parallel, sglang_max_workers, sglang_temperature],
                      outputs=output,
                      queue=True)
        demo.queue()

    return demo

if __name__ == "__main__":
    create_ui().launch()
