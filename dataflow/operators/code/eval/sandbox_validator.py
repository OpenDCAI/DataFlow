import pandas as pd
from typing import List, Tuple

# Assuming these are the correct import paths for your framework
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC

# Import the provided executor script
# Assuming shared_vis_python_exe.py is in the same directory or accessible via PYTHONPATH
from .shared_vis_python_exe import PythonExecutor

@OPERATOR_REGISTRY.register()
class SandboxValidator(OperatorABC):
    """
    SandboxValidator is an operator that executes code snippets in a secure,
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
        # Initialize the PythonExecutor here. It will be reused for all code snippets.
        self.executor = PythonExecutor(
            get_answer_from_stdout=True,  # Capture print statements as primary output
            timeout_length=timeout_length,
            use_process_isolation=use_process_isolation
        )
        self.logger.info(f"SandboxValidator initialized with timeout={timeout_length}s and process_isolation={use_process_isolation}")
    
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
        required_keys = [self.input_code_key]
        forbidden_keys = [self.output_status_key, self.output_log_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s) for SandboxValidator: {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten by SandboxValidator: {conflict}")
    
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

    def run(
        self, 
        storage: DataFlowStorage, 
        input_code_key: str = "generated_code",
        output_status_key: str = "sandbox_status",
        output_log_key: str = "sandbox_log"
    ) -> List[str]:
        """
        Executes the sandbox validation process.
        
        It reads data, executes each code snippet, captures the status and log,
        and writes the results back to storage.
        
        Returns:
            A list containing the names of the newly created output columns.
        """
        self.logger.info("Running SandboxValidator operator...")
        
        # Store keys for use in helper methods
        self.input_code_key = input_code_key
        self.output_status_key = output_status_key
        self.output_log_key = output_log_key

        # 1. Read data from the current step
        dataframe = storage.read("dataframe")
        if dataframe.empty:
            self.logger.warning("Input dataframe for SandboxValidator is empty. Skipping.")
            storage.write(dataframe)
            return []
        
        # 2. Validate the data
        self._validate_dataframe(dataframe)
        
        # 3. Prepare the batch of code to execute
        code_to_run = dataframe[self.input_code_key].tolist()
        self.logger.info(f"Preparing to execute {len(code_to_run)} code snippets in the sandbox...")
        
        # 4. Execute the code batch
        execution_results = self._execute_code_batch(code_to_run)
        
        # 5. Unpack results and add them to the DataFrame
        statuses, logs = zip(*execution_results)
        dataframe[self.output_status_key] = statuses
        dataframe[self.output_log_key] = logs
        
        # 6. Write the final results back to storage
        output_file = storage.write(dataframe)
        pass_count = statuses.count("PASS")
        self.logger.success(f"SandboxValidator finished. {pass_count}/{len(code_to_run)} snippets passed. Results saved to {output_file}")

        # 7. Return the names of the new columns
        return [self.output_status_key, self.output_log_key]

    def __del__(self):
        """
        Ensures the executor's resources are cleaned up when the operator is destroyed.
        """
        if hasattr(self, 'executor') and self.executor:
            # The executor's __del__ method handles terminating the worker process.
            del self.executor