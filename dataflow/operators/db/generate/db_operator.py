from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import MyScaleDBStorage

@OPERATOR_REGISTRY.register()
class DBOperator(OperatorABC):
    def __init__(self, expr):
        """
        Initialize the DBOperator with the provided expression.
        
        Args:
            expr (str): The SQL expression to execute.
        """
        self.logger = get_logger()
        self.expr = expr
    
    def run(self, storage:MyScaleDBStorage, input_key:str) -> list:
        """
        Execute the SQL expression against the database storage.
        
        Args:
            storage (DBStorage): The database storage instance to use.
            input_key (str): The key for the input data.
        
        Returns:
            list: The result of the SQL query execution.
        """
        self.logger.info(f"Executing SQL expression: {self.expr}")
        result = storage.execute(self.expr, input_key)
        self.logger.info(f"Query executed successfully, retrieved {len(result)} records.")
        return result