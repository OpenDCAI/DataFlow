from dataflow.operators.text2sql import (
    Text2SQLPromptGenerator
)
from dataflow.operators.text2sql import (
    SQLExecutionFilter
)
from dataflow.operators.text2sql import (
    SQLComponentClassifier
)
from dataflow.utils.storage import FileStorage
from dataflow.utils.text2sql.database_manager import DatabaseManager


class Text2SQL_CPUPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/Text2SQLPipeline/pipeline_refine.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # You can customize the difficulty config here, but it must contain 'thresholds' and 'labels' keys
        component_difficulty_config = {
            'thresholds': [2, 4, 6],      
            'labels': ['easy', 'medium', 'hard', 'extra']
        }

        # You can customize the prompt template here, but it must contain {schema} and {question} placeholders
        prompt_template = '''Task Overview:
            /* Given the following database schema: */
            {schema}
            /* Answer the following: {question} */
            Let's think step by step'''

        # You can customize the schema config here, but it must contain 'format' and 'use_example' keys
        schema_config = {
            'format': 'ddl',  # Optional: 'ddl', 'formatted_schema'
            'use_example': True  # Whether to include example data
        }

        # A demo database is provided. Download it from the following URL and update the path:  
        # https://huggingface.co/datasets/Open-Dataflow/dataflow-Text2SQL-database-example  
        db_root_path = ""

        # SQL execution timeout. Generated SQL execution time should be less than this value.
        sql_execution_timeout = 2

        # SQLite and MySQL are currently supported
        # db_type can be sqlite or mysql, which must match your database type
        # If sqlite is selected, root_path must be provided, this path must exist and contain database files
        # If mysql is selected, host, user, password must be provided, these credentials must be correct and have access permissions
        # MySQL example:
        # database_manager = DatabaseManager(
        #     db_type="mysql",
        #     config={
        #         "host": "localhost",
        #         "user": "root",
        #         "password": "your_password",
        #         "database": "your_database_name"
        #     }
        # )
        # SQLite example:
        database_manager = DatabaseManager(
            db_type="sqlite",
            config={
                "root_path": db_root_path
            },
            logger=None,
            max_connections_per_db=100,
            max_workers=100
        )

        self.sql_execution_filter_step1 = SQLExecutionFilter(
            database_manager=database_manager,
            timeout=sql_execution_timeout
        )

        self.text2sql_prompt_generator_step2 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=prompt_template,
            schema_config=schema_config
        )

        self.sql_component_classifier_step3 = SQLComponentClassifier(
            difficulty_config=component_difficulty_config
        )
        
        
    def forward(self):

        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"

        self.sql_execution_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.text2sql_prompt_generator_step2.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            output_prompt_key="prompt"
        )

        self.sql_component_classifier_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            output_difficulty_key="sql_component_difficulty"
        )


if __name__ == "__main__":
    model = Text2SQL_CPUPipeline()
    model.forward()

