import os
import json
import argparse
import importlib
import inspect
import logging
import traceback
from pathlib import Path
from io import StringIO
from typing import List, Tuple, Dict, Optional
import sys


CURRENT_SCRIPT_PATH = Path(__file__).resolve()  
TEST_DIR = CURRENT_SCRIPT_PATH.parent  
PROJECT_ROOT = TEST_DIR.parent  


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("PipelineRunner") 



def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with support for custom Pipeline directories."""
    parser = argparse.ArgumentParser(
        description="Execute API Pipeline in batches, automatically switch the working directory, and record a complete log.",
        formatter_class=argparse.RawTextHelpFormatter  # 支持换行显示帮助信息
    )
    # 默认Pipeline目录：项目根/run_dataflow/api_pipelines
    default_pipeline_dir = PROJECT_ROOT / "run_dataflow" / "api_pipelines"
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default=str(default_pipeline_dir.resolve()),
        help=f"API Pipeline absolute path\nexample：/home/user/project/run_dataflow/api_pipelines\ndefault：{default_pipeline_dir.resolve()}"
    )
    return parser.parse_args()


def switch_working_directory(target_dir: Path) -> Optional[str]:
    """Switch to the target working directory and return the original directory for subsequent recovery.
    If the switch fails, raise an exception and terminate the script.
    """
    if not target_dir.exists():
        raise FileNotFoundError(f"Pipeline directory does not exist: {target_dir}")
    if not target_dir.is_dir():
        raise NotADirectoryError(f"The specified path is not a directory: {target_dir}")
    
    original_dir = os.getcwd()
    try:
        os.chdir(str(target_dir))
        logger.info(f"Working directory switched successfully: [{original_dir}] → [{os.getcwd()}]")
        return original_dir
    except Exception as e:
        raise RuntimeError(f"Failed to switch working directory: {str(e)}") from e


def collect_pipeline_classes(pipeline_dir: Path) -> Tuple[List[Tuple[str, type, str]], List[Dict]]:
    """Collect all custom Pipeline classes in the specified directory.
    Returns: (successfully collected class list, import error detail list)
    Class list format: (class name, class object, file name)
    Error list format: {file name, error type, error message, stack trace}
    """
    pipeline_classes: List[Tuple[str, type, str]] = []
    import_errors: List[Dict] = []
    current_script_name = CURRENT_SCRIPT_PATH.name  # skip self script
    

    for py_file in pipeline_dir.glob("*.py"):
        if py_file.name == current_script_name:
            continue
        
        try:
            # construct module path (based on project root, ensure correct import)
            relative_path = py_file.relative_to(PROJECT_ROOT)
            module_name = ".".join(relative_path.with_suffix("").parts)
            
            # import module and filter "classes defined in the current file" (exclude external imported classes)
            module = importlib.import_module(module_name)
            for class_name, cls in inspect.getmembers(module, inspect.isclass):
                if cls.__module__ == module_name:  # ensure the class is defined in the current file
                    pipeline_classes.append((class_name, cls, py_file.name))
                    logger.info(f"发现Pipeline类：{class_name}（文件：{py_file.name}）")
        
        except Exception as e:
            # record complete error information (including stack trace) for troubleshooting
            error_detail = {
                "file_name": py_file.name,
                "error_type": type(e).__name__,
                "error_msg": str(e),
                "stack_trace": traceback.format_exc()  # complete stack trace, more detailed than inspect
            }
            import_errors.append(error_detail)
            logger.error(f"Failed to import file {py_file.name}: {str(e)}")
    
    return pipeline_classes, import_errors


def run_single_pipeline(pipeline_class: type, class_name: str, file_name: str) -> Dict:
    """
    Run a single Pipeline and return the execution result (including error details).
    New: Preserve terminal output when redirecting, implement "cache record + terminal display" dual output.
    """
    result = {
        "pipeline_class": class_name,
        "file_name": file_name,
        "status": "success",
        "error": None,
        "error_details": None
    }

    # ------------------------------
    # Core modification: custom dual output stream (write cache and terminal at the same time)
    # ------------------------------
    class TeeStringIO(StringIO):
        def __init__(self, original_stream):
            super().__init__()
            self.original_stream = original_stream  # save the original terminal stream (for real-time output)

        def write(self, s):
            super().write(s)  # write content to cache (for subsequent error detection)
            self.original_stream.write(s)  # write content to terminal (preserve real-time display)   
            self.original_stream.flush()  
            self.original_stream.flush()  
        def flush(self):
            super().flush()
            self.original_stream.flush()  

    # initialize output capture (stdout + stderr use dual output stream)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    # use custom TeeStringIO, implement "cache + terminal" dual output
    captured_print = TeeStringIO(old_stdout)
    captured_stderr = TeeStringIO(old_stderr)
    sys.stdout = captured_print
    sys.stderr = captured_stderr

    # custom Logging Handler: capture the output of the root logger (e.g. ERROR:root:xxx)
    class CaptureLogHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.logs = []

        def emit(self, record: logging.LogRecord) -> None:
            log_msg = self.format(record)
            self.logs.append(log_msg)  # record to cache
            # logging already outputs to terminal by default, no additional processing needed

    log_handler = CaptureLogHandler()
    log_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)

    try:        
        logger.info(f"Start running Pipeline: {class_name} (file: {file_name})")    
        # initialize Pipeline
        pipeline = pipeline_class()

        # configure LLM Serving (avoid duplicate code)
        if hasattr(pipeline, "llm_serving"):
            llm_config = {
                "api_url": "http://123.129.219.111:3000/v1/chat/completions",
                "model_name": "gpt-3.5-turbo",
                "max_workers": 15
            }
            for key, value in llm_config.items():
                setattr(pipeline.llm_serving, key, value)
            logger.debug(f"LLM Serving configured: {llm_config}")

        # execute Pipeline (support forward/run two methods)
        output = None
        if hasattr(pipeline, "forward"):
            output = pipeline.forward()
        elif hasattr(pipeline, "run"):
            output = pipeline.run()
        else:
            raise AttributeError("No forward or run method found, cannot execute Pipeline")


        # merge all outputs (stdout/stderr/logging all included)
        full_output = "\n".join([
            "=== Captured Print Output ===",
            captured_print.getvalue(),
            "=== Captured STDERR Output ===",
            captured_stderr.getvalue(),
            "=== Captured Logging Output ===",
            "\n".join(log_handler.logs),
            "=== Pipeline Return Value ===",
            str(output) if output is not None else "No return value"
        ])

        # ------------------------------
            # Core modification: detect error in stdout/stderr (case-insensitive)
        # ------------------------------
        # 1. detect error in stdout/stderr (cover all lowercase/uppercase/mixed case)
        if "error" in full_output.lower():
            # extract the context containing error,便于定位问题
            error_lines = [line for line in full_output.split("\n") if "error" in line.lower()]
            error_context = "\n".join(error_lines[:5])  # extract the first 5 lines of the context
            raise RuntimeError(f"Detected error in stdout/stderr, context: \n{error_context}")

        # 2. detect if there is an ERROR level log (e.g. logging.error() output log)
        for log in log_handler.logs:
            if log.startswith("ERROR:"):
                raise RuntimeError(f"Detected ERROR level log: {log}")

        logger.info(f"Pipeline executed successfully: {class_name}")
        return result

    except Exception as e:
        # record error details (including complete context)
        full_output = "\n".join([
            "=== Captured Print Output ===",
            captured_print.getvalue(),
            "=== Captured STDERR Output ===",
            captured_stderr.getvalue(),
            "=== Captured Logging Output ===",
            "\n".join(log_handler.logs),
            "=== Pipeline Return Value ===",
            str(output) if 'output' in locals() else "No return value"
        ])
        error_msg = f"{type(e).__name__}: {str(e)}"
        error_details = f"{traceback.format_exc()}\n\n=== Complete Execution Context ===\n{full_output}"

        result.update({
            "status": "failed",
            "error": error_msg,
            "error_details": error_details
        })
        logger.error(f"Pipeline execution failed: {error_msg}", exc_info=True)
        return result

    finally:
        # restore the original output stream, avoid affecting subsequent logic
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if log_handler in root_logger.handlers:
            root_logger.removeHandler(log_handler)


# ------------------------------
# 6. export logs (optimize format, distinguish import errors and run errors)
# ------------------------------
def export_run_logs(
    run_results: List[Dict],
    import_errors: List[Dict],
    log_dir: Path = TEST_DIR / "pipeline_logs"
) -> None:
    """
    Export run logs:
    1. JSON file: complete run results + import errors (for debugging)
    2. TXT log: key error information (for quick viewing)
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = os.popen('date "+%Y%m%d_%H%M%S"').read().strip()  # timestamp, avoid log overwrite
    
    # 1. complete JSON log (including all details)
    json_data = {
        "run_time": current_time,
        "pipeline_dir": str(PIPELINE_DIR),
        "import_errors": import_errors,
        "run_results": run_results
    }
    json_path = log_dir / f"pipeline_full_results_{current_time}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # 2. simplified TXT log (key information, for quick viewing)
    log_path = log_dir / f"pipeline_error_summary_{current_time}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"===== Pipeline Batch Running Log ({current_time}) =====\n")
        f.write(f"Pipeline Directory: {PIPELINE_DIR}\n")
        f.write(f"Total Detected Files: {len(import_errors) + len(run_results)}\n")
        f.write(f"Import Errors: {len(import_errors)}\n")
        f.write(f"Success: {sum(1 for res in run_results if res['status'] == 'success')}\n")
        f.write(f"Failed: {sum(1 for res in run_results if res['status'] == 'failed')}\n\n")
        
        # write import errors
        if import_errors:
            f.write("===== Import Errors Details =====\n")
            for idx, err in enumerate(import_errors, 1):
                f.write(f"[{idx}] File: {err['file_name']}\n")
                f.write(f"    Error Type: {err['error_type']}\n")
                f.write(f"    Error Message: {err['error_msg']}\n")
                f.write(f"    Complete Stack Trace: see JSON log import_errors[{idx-1}].stack_trace\n\n")
        
        # write run errors
        run_errors = [res for res in run_results if res['status'] == 'failed']
        if run_errors:
            f.write("===== Run Errors Details =====\n")
            for idx, err in enumerate(run_errors, 1):
                f.write(f"[{idx}] Class Name: {err['pipeline_class']} (File: {err['file_name']})\n")
                f.write(f"    Error Message: {err['error']}\n")
                f.write(f"    Complete Details: see JSON log run_results[{run_results.index(err)}].error_details\n\n")
    
    logger.info(f"Logs exported to: {log_dir}")
    logger.info(f"Complete JSON Log: {json_path.name}")
    logger.info(f"Simplified Error Log: {log_path.name}")


# ------------------------------
# 7. main function (process chain, more robust exception capture)
# ------------------------------
def main() -> None:
    try:
        # step 1: parse arguments
        args = parse_arguments()
        global PIPELINE_DIR  # global variable, for log export usage
        PIPELINE_DIR = Path(args.pipeline_dir).resolve()
        
        # step 2: add project root to Python path (ensure import dataflow)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))
            logger.debug(f"Added project root to Python path: {PROJECT_ROOT}")
        
        # step 3: switch working directory
        original_dir = switch_working_directory(PIPELINE_DIR)
        
        # step 4: collect Pipeline classes (including import errors)
        pipeline_classes, import_errors = collect_pipeline_classes(PIPELINE_DIR)
        if not pipeline_classes and not import_errors:
            logger.warning("No .py files found, no need to run")
            return
        if not pipeline_classes:
            logger.error("No runnable Pipeline classes found, exiting")
            export_run_logs([], import_errors)  # export import errors log
            return
        
        # step 5: batch run Pipeline
        logger.info(f"Start batch running, total {len(pipeline_classes)} Pipelines")
        run_results = []
        for class_name, cls, file_name in pipeline_classes:
            run_res = run_single_pipeline(cls, class_name, file_name)
            run_results.append(run_res)
        
        # step 6: export logs
        export_run_logs(run_results, import_errors)
        
        # step 7: output run summary
        success_count = sum(1 for res in run_results if res['status'] == 'success')
        fail_count = len(run_results) - success_count
        print(f"\n===== Batch Running Completed =====\n")
        print(f"📊 Run Summary:")
        print(f"   Total Pipelines: {len(run_results)}")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failed: {fail_count}")
        print(f"   ⚠️  Import Errors: {len(import_errors)}")
        print(f"\nLog Location: {TEST_DIR / 'pipeline_logs'}")
        
    except Exception as e:
        # capture script-level exceptions (e.g. directory switch failure, parameter error)
        logger.error(f"Script execution failed: {str(e)}", exc_info=True)
        print(f"\n❌ Script execution error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
