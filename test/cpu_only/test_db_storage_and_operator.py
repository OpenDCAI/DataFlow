from dataflow.operators.db.db_operator import DBOperator

from dataflow.utils.storage import MyScaleDBStorage
from dataflow.operators.generate import QuestionDifficultyClassifier
class DBShowCasePipeline():
    def __init__(self):
        self.storage = MyScaleDBStorage()
        self.operator_step1 = DBOperator("SELECT * FROM example_table")
        self.operator_step2 = QuestionDifficultyClassifier()
    def run(self):
        """
        Run the DBOperator with the DBStorage.
        """

        # integrate the DBOperator with common Dataflow operators
        output_keys_step1 = self.operator_step1.run(
            storage=self.storage, 
            input_key="example_input"
        )
        print(f"Operation finished. Output keys from DBOperator: {output_keys_step1}")


        output_keys_step2 = self.operator_step2.run(
            storage=self.storage,
            input_key="example_input"
        )
        print(f"Operation finished. Output keys from QuestionDifficultyClassifier: {output_keys_step2}")


        
