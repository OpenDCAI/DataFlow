import pandas as pd
from typing import List, Tuple

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

# Import the provided executor script
# Assuming shared_vis_python_exe.py is in the same directory or accessible via PYTHONPATH
from .python_executor import PythonExecutor

@OPERATOR_REGISTRY.register()
class CodeSandboxSampleEvaluator(OperatorABC):
    """
    CodeSandboxSampleEvaluator is an operator that executes code snippets in a secure,
    isolated environment to verify their correctness. It leverages a robust
    PythonExecutor to handle process isolation, timeouts, and capturing results.
    This is the final validation step in the data synthesis pipeline.
    """

    def __init__(self, language: str = "python", timeout_length: int = 15, use_process_isolation: bool = True):
        """
        Initializes the operator and the underlying PythonExecutor.
        
        Args:
            timeout_length: Maximum execution time in seconds for each code snippet.
            use_process_isolation: Whether to run code in a separate process for security. Recommended to keep True.
        """
        self.logger = get_logger()
        self.logger.info(f'Initializing {self.__class__.__name__}...')
        # Initialize the PythonExecutor here. It will be reused for all code snippets.
        self.executor = PythonExecutor(
            get_answer_from_stdout=True,  # Capture print statements as primary output
            timeout_length=timeout_length,
            use_process_isolation=use_process_isolation
        )
        self.score_name = 'SandboxValidationScore'
        self.logger.info(f'{self.__class__.__name__} initialized.')
    
    @staticmethod
    def get_desc(lang: str = "en"):
        """
        Provides a description of the operator's function and parameters.
        """
        if lang == "zh":
            return (
                "该算子在一个安全的沙箱环境中执行代码片段以验证其正确性。\n\n"
                "输入参数：\n"
                "- input_code_key: 包含待执行代码的字段名 (默认: 'generated_code')\n"
                "输出参数：\n"
                "- output_status_key: 用于存储执行状态 ('PASS' 或 'FAIL') 的字段名 (默认: 'sandbox_status')\n"
                "- output_log_key: 用于存储执行日志或错误信息的字段名 (默认: 'sandbox_log')\n"
            )
        else: # Default to English
            return (
                "This operator executes code snippets in a secure sandbox environment to verify their correctness.\n\n"
                "Input Parameters:\n"
                "- input_code_key: Field name containing the code to be executed (default: 'generated_code')\n"
                "Output Parameters:\n"
                "- output_status_key: Field name to store the execution status ('PASS' or 'FAIL') (default: 'sandbox_status')\n"
                "- output_log_key: Field name to store the execution log or error message (default: 'sandbox_log')\n"
            )

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        """
        Validates the DataFrame to ensure the required code column exists and output columns don't.
        """
        required_keys = [self.input_key]
        forbidden_keys = [self.output_status_key, self.output_log_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for CodeSandboxSampleEvaluator: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by CodeSandboxSampleEvaluator: {conflict}")
    
    def _score_func(self, code: str) -> Tuple[str, str]:
        """
        Execute a single code snippet and return status and log.
        
        Args:
            code: Code snippet to execute
            
        Returns:
            Tuple of (status, log) where status is 'PASS' or 'FAIL'
        """
        try:
            result, report = self.executor.execute([code], messages=[])
            
            if report == "Done":
                status = "PASS"
                log = result.get('text', '') if isinstance(result, dict) else str(result)
            else:
                status = "FAIL"
                log = report
                
            return status, log
        except Exception as e:
            return "FAIL", f"Execution error: {str(e)}"
    
    def _execute_code_batch(self, code_list: List[str]) -> List[Tuple[str, str]]:
        """
        Execute a batch of code snippets using the PythonExecutor.
        
        Args:
            code_list: A list of strings, where each string is a code snippet.
            
        Returns:
            A list of tuples, where each tuple contains (status, log).
            Status can be 'PASS' or 'FAIL', log contains execution output or error message.
        """
        # The executor's batch_apply takes a list of code strings and a 'messages' context.
        # For our simple validation, the context can be an empty list.
        results_with_reports = self.executor.batch_apply(code_list, messages=[])
        
        processed_results = []
        for (result, report) in results_with_reports:
            # The executor's report tells us about success or failure.
            # "Done" indicates success. Anything else (e.g., "Error: ...", "Timeout Error") indicates failure.
            if report == "Done":
                status = "PASS"
                # The 'result' can be a dict with 'text' and/or 'images'. We just need the text log.
                log = result.get('text', '') if isinstance(result, dict) else result
            else:
                status = "FAIL"
                # The report itself is the most informative log on failure.
                log = report
            
            processed_results.append((status, log))
            
        return processed_results

    def eval(self, dataframe: pd.DataFrame, input_key: str) -> Tuple[List[str], List[str]]:
        """
        Execute code snippets and return statuses and logs.
        
        Args:
            dataframe: Input DataFrame
            input_key: Field name containing code snippets
            
        Returns:
            Tuple of (statuses, logs) lists
        """
        self.logger.info(f"Evaluating {self.score_name}...")
        
        code_list = dataframe[input_key].tolist()
        execution_results = self._execute_code_batch(code_list)
        
        statuses, logs = zip(*execution_results)
        self.logger.info("Evaluation complete!")
        return list(statuses), list(logs)
    
    def run(
        self, 
        storage: DataFlowStorage, 
        input_key: str,
        output_status_key: str = "sandbox_status",
        output_log_key: str = "sandbox_log"
    ):
        """
        Executes the sandbox validation process.
        
        Args:
            storage: Data storage object
            input_key: Field name containing code snippets
            output_status_key: Field name for execution status
            output_log_key: Field name for execution logs
        """
        self.input_key = input_key
        self.output_status_key = output_status_key
        self.output_log_key = output_log_key
        
        dataframe = storage.read("dataframe")
        statuses, logs = self.eval(dataframe, input_key)
        
        dataframe[self.output_status_key] = statuses
        dataframe[self.output_log_key] = logs
        storage.write(dataframe)

    def __del__(self):
        """
        Ensures the executor's resources are cleaned up when the operator is destroyed.
        """
        if hasattr(self, 'executor') and self.executor:
            # The executor's __del__ method handles terminating the worker process.
            del self.executor