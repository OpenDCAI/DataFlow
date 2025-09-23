# This code is referenced from PyVision: Agentic Vision with Dynamic Tooling
# GitHub: https://github.com/agents-x-project/PyVision
# Paper: PyVision: Agentic Vision with Dynamic Tooling (arXiv:2507.07998)

import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocessing
from multiprocessing import Queue, Process
from typing import Any, Dict, Optional, Tuple, List, Union
from tqdm import tqdm
from concurrent.futures import TimeoutError
from contextlib import redirect_stdout
import base64
from io import BytesIO
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import queue

def encode_image(image_path):
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(
    base64_str: str, 
    remove_prefix: bool = True, 
    convert_mode: Optional[str] = "RGB"
) -> Union[Image.Image, None]:
    """
    Convert a Base64-encoded image string to a PIL Image object.

    Args:
        base64_str: Base64-encoded image string (can include data: prefix)
        remove_prefix: Whether to automatically remove the "data:image/..." prefix (default True)
        convert_mode: Convert to the specified mode (such as "RGB"/"RGBA", None means no conversion)

    Returns:
        PIL.Image.Image object, or None if decoding fails
        
    Examples:
        >>> img = base64_to_image("data:image/png;base64,iVBORw0KGg...")
        >>> img = base64_to_image("iVBORw0KGg...", remove_prefix=False)
    """
    try:
        # 1. Handle Base64 prefix
        if remove_prefix and "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # 2. Decode Base64
        image_data = base64.b64decode(base64_str)
        
        # 3. Convert to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # 4. Optional mode conversion
        if convert_mode:
            image = image.convert(convert_mode)
            
        return image
    
    except (base64.binascii.Error, OSError, Exception) as e:
        print(f"Base64 decode failed: {str(e)}")
        return None


class PersistentWorker:
    """Persistent worker process."""
    
    def __init__(self):
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.start()
    
    def start(self):
        """Start the worker process."""
        self.process = Process(target=self._worker_loop)
        self.process.daemon = True
        self.process.start()
    
    def _worker_loop(self):
        """Main loop for the worker process."""
        runtime = None
        runtime_class = None
        
        while True:
            try:
                # Get task
                task = self.input_queue.get()
                
                if task is None:  # Termination signal
                    break
                
                task_type = task.get('type')
                
                if task_type == 'init':
                    # Initialize runtime
                    messages = task.get('messages', [])
                    runtime_class = task.get('runtime_class', ImageRuntime)
                    runtime = runtime_class(messages)
                    self.output_queue.put({
                        'status': 'success',
                        'result': 'Initialized'
                    })
                
                elif task_type == 'execute':
                    # Execute code
                    if runtime is None:
                        messages = task.get('messages', [])
                        runtime_class = task.get('runtime_class', ImageRuntime)
                        runtime = runtime_class(messages)
                    
                    code = task.get('code')
                    get_answer_from_stdout = task.get('get_answer_from_stdout', True)
                    answer_symbol = task.get('answer_symbol')
                    answer_expr = task.get('answer_expr')
                    
                    try:
                        # Record the number of images before execution
                        pre_figures_count = len(runtime._global_vars.get("_captured_figures", []))
                        
                        if get_answer_from_stdout:
                            program_io = io.StringIO()
                            with redirect_stdout(program_io):
                                runtime.exec_code("\n".join(code))
                            program_io.seek(0)
                            result = program_io.read()
                        elif answer_symbol:
                            runtime.exec_code("\n".join(code))
                            result = runtime._global_vars.get(answer_symbol, "")
                        elif answer_expr:
                            runtime.exec_code("\n".join(code))
                            result = runtime.eval_code(answer_expr)
                        else:
                            if len(code) > 1:
                                runtime.exec_code("\n".join(code[:-1]))
                                result = runtime.eval_code(code[-1])
                            else:
                                runtime.exec_code("\n".join(code))
                                result = ""
                        
                        # Get newly generated images
                        all_figures = runtime._global_vars.get("_captured_figures", [])
                        new_figures = all_figures[pre_figures_count:]
                        
                        # Build result
                        if new_figures:
                            result = {
                                'text': result,
                                'images': new_figures
                            } if result else {'images': new_figures}
                        else:
                            result = {'text': result} if result else {}
                        
                        self.output_queue.put({
                            'status': 'success',
                            'result': result,
                            'report': 'Done'
                        })
                    
                    except Exception as e:
                        self.output_queue.put({
                            'status': 'error',
                            'error': str(e),
                            'traceback': traceback.format_exc(),
                            'report': f'Error: {str(e)}'
                        })
                
                elif task_type == 'reset':
                    # Reset runtime
                    messages = task.get('messages', [])
                    runtime_class = task.get('runtime_class', ImageRuntime)
                    runtime = runtime_class(messages)
                    self.output_queue.put({
                        'status': 'success',
                        'result': 'Reset'
                    })
                    
            except Exception as e:
                self.output_queue.put({
                    'status': 'error',
                    'error': f'Worker error: {str(e)}',
                    'traceback': traceback.format_exc()
                })
    
    def execute(self, code: List[str], messages: list = None, runtime_class=None, 
                get_answer_from_stdout=True, answer_symbol=None, answer_expr=None, timeout: int = 30):
        """Execute code."""
        self.input_queue.put({
            'type': 'execute',
            'code': code,
            'messages': messages,
            'runtime_class': runtime_class,
            'get_answer_from_stdout': get_answer_from_stdout,
            'answer_symbol': answer_symbol,
            'answer_expr': answer_expr
        })
        
        try:
            result = self.output_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return {
                'status': 'error',
                'error': 'Execution timeout',
                'report': 'Timeout Error'
            }
    
    def init_runtime(self, messages: list, runtime_class=None):
        """Initialize runtime."""
        self.input_queue.put({
            'type': 'init',
            'messages': messages,
            'runtime_class': runtime_class
        })
        return self.output_queue.get()
    
    def reset_runtime(self, messages: list = None, runtime_class=None):
        """Reset runtime."""
        self.input_queue.put({
            'type': 'reset',
            'messages': messages,
            'runtime_class': runtime_class
        })
        return self.output_queue.get()
    
    def terminate(self):
        """Terminate the worker process."""
        if self.process and self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        self._captured_figures = []

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        # Security check
        if regex.search(r"(\s|^)?(input|os\.system|subprocess)\(", code_piece):
            raise RuntimeError("Forbidden function calls detected")
        
        # Detect and modify plt.show() calls
        if "plt.show()" in code_piece:
            modified_code = code_piece.replace("plt.show()", """
# Capture current figure
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
_captured_image = base64.b64encode(buf.read()).decode('utf-8')
_captured_figures.append(_captured_image)
plt.close()
""")
            # Ensure _captured_figures variable exists
            if "_captured_figures" not in self._global_vars:
                self._global_vars["_captured_figures"] = []
            
            exec(modified_code, self._global_vars)
        else:
            exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars.get("answer", None)
    
    @property
    def captured_figures(self):
        return self._global_vars.get("_captured_figures", [])


class ImageRuntime(GenericRuntime):
    HEADERS = [
        "import matplotlib",
        "matplotlib.use('Agg')",  # Use non-interactive backend
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "import io",
        "import base64",
        "import numpy as np",
        "_captured_figures = []",  # Initialize image capture list
    ]

    def __init__(self, messages):
        super().__init__()

        image_var_dict = {}
        image_var_idx = 0
        init_captured_figures = []
        
        for message_item in messages:
            content = message_item['content']
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get('type')
                    if item_type == "image_url":
                        item_image_url = item['image_url']['url']
                        image = base64_to_image(item_image_url)
                        if image:
                            image_var_dict[f"image_clue_{image_var_idx}"] = image
                            init_captured_figures.append(base64.b64encode(
                                BytesIO(image.tobytes()).getvalue()).decode('utf-8'))
                            image_var_idx += 1

        image_var_dict["_captured_figures"] = init_captured_figures
        self.inject(image_var_dict)


class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {}
    HEADERS = [
        "import datetime",
        "from dateutil.relativedelta import relativedelta",
        "timedelta = relativedelta"
    ]


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime_class=None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = True,
        timeout_length: int = 20,
        use_process_isolation: bool = True,
    ) -> None:
        self.runtime_class = runtime_class if runtime_class else ImageRuntime
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length
        self.use_process_isolation = use_process_isolation
        self.persistent_worker = None

    def _ensure_worker(self):
        """Ensure the worker process exists."""
        if self.persistent_worker is None:
            self.persistent_worker = PersistentWorker()

    def process_generation_to_code(self, gens: str):
        return [g.split("\n") for g in gens]

    def execute(
        self,
        code,
        messages,
        get_answer_from_stdout=True,
        runtime_class=None,
        answer_symbol=None,
        answer_expr=None,
    ) -> Tuple[Union[str, Dict[str, Any]], str]:
        
        if self.use_process_isolation:
            # Ensure worker process exists
            self._ensure_worker()
            
            # Execute code
            result = self.persistent_worker.execute(
                code,
                messages,
                runtime_class or self.runtime_class,
                get_answer_from_stdout,
                answer_symbol,
                answer_expr,
                timeout=self.timeout_length
            )
            
            if result['status'] == 'success':
                return result['result'], result.get('report', 'Done')
            else:
                error_result = {
                    'error': result.get('error', 'Unknown error'),
                    'traceback': result.get('traceback', '')
                }
                return error_result, result.get('report', f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Non-isolation mode (for backward compatibility)
            runtime = runtime_class(messages) if runtime_class else self.runtime_class(messages)
            
            try:
                if get_answer_from_stdout:
                    program_io = io.StringIO()
                    with redirect_stdout(program_io):
                        runtime.exec_code("\n".join(code))
                    program_io.seek(0)
                    result = program_io.read()
                elif answer_symbol:
                    runtime.exec_code("\n".join(code))
                    result = runtime._global_vars.get(answer_symbol, "")
                elif answer_expr:
                    runtime.exec_code("\n".join(code))
                    result = runtime.eval_code(answer_expr)
                else:
                    if len(code) > 1:
                        runtime.exec_code("\n".join(code[:-1]))
                        result = runtime.eval_code(code[-1])
                    else:
                        runtime.exec_code("\n".join(code))
                        result = ""

                # Check for captured figures
                captured_figures = runtime.captured_figures
                if captured_figures:
                    result = {
                        'text': result,
                        'images': captured_figures
                    } if result else {'images': captured_figures}
                else:
                    result = {'text': result} if result else {}

                report = "Done"

            except Exception as e:
                result = {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                report = f"Error: {str(e)}"

            return result, report

    def apply(self, code, messages):
        return self.batch_apply([code], messages)[0]

    @staticmethod
    def truncate(s, max_length=400):
        if isinstance(s, dict):
            # If it is a dict (with images), truncate only the text part
            if 'text' in s:
                half = max_length // 2
                if len(s['text']) > max_length:
                    s['text'] = s['text'][:half] + "..." + s['text'][-half:]
            return s
        else:
            half = max_length // 2
            if isinstance(s, str) and len(s) > max_length:
                s = s[:half] + "..." + s[-half:]
            return s

    def batch_apply(self, batch_code, messages):
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []

        if len(all_code_snippets) > 100:
            progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
        else:
            progress_bar = None

        for code in all_code_snippets:
            try:
                result = self.execute(
                    code,
                    messages=messages,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime_class=self.runtime_class,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                )
                all_exec_results.append(result)
            except TimeoutError as error:
                print(error)
                all_exec_results.append(("", "Timeout Error"))
                timeout_cnt += 1
            except Exception as error:
                print(f"Error in batch_apply: {error}")
                all_exec_results.append(("", f"Error: {str(error)}"))
            
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # Handle results
            if isinstance(res, dict):
                # If result contains images, special handling
                if 'text' in res:
                    res['text'] = str(res['text']).strip()
                    res['text'] = self.truncate(res['text'])
                report = str(report).strip()
                report = self.truncate(report)
            else:
                # Normal text result
                res = str(res).strip()
                res = self.truncate(res)
                report = str(report).strip()
                report = self.truncate(report)
            batch_results.append((res, report))
        return batch_results

    def reset(self, messages=None):
        """Reset executor state."""
        if self.use_process_isolation and self.persistent_worker:
            self.persistent_worker.reset_runtime(messages, self.runtime_class)

    def __del__(self):
        """Clean up resources."""
        if self.persistent_worker:
            self.persistent_worker.terminate()