from dataflow.operators.text2sql import (
    SQLGenerator,
    Text2SQLQuestionGenerator,
    Text2SQLPromptGenerator,
    Text2SQLCoTGenerator
)
from dataflow.operators.text2sql import (
    SQLExecutionFilter
)
from dataflow.operators.text2sql import (
    SQLComponentClassifier,
    SQLExecutionClassifier
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm
from dataflow.utils.text2sql.database_manager import DatabaseManager


class Text2SQLGeneration_APIPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving = APILLMServing_request(
            api_url="http://api.openai.com/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )

        # It is recommended to use better LLMs for the generation of Chain-of-Thought (CoT) reasoning process.
        cot_generation_api_llm_serving = APILLMServing_request(
            api_url="http://api.openai.com/v1/chat/completions",
            model_name="gpt-4o", # You can change to a more powerful model for CoT generation
            max_workers=100
        )

        embedding_serving = APILLMServing_request(
            api_url="http://api.openai.com/v1/embeddings",
            model_name="text-embedding-ada-002",
            max_workers=100
        )

        # You can customize the difficulty config here, but it must contain 'num_generations', 'thresholds' and 'labels' keys
        # 'num_generations' key is the number of generations for each question, SQL will be classified based on the number of correct executions
        execution_difficulty_config = {
            "num_generations": 10,
            'thresholds': [2, 5, 9],
            'labels': ['extra', 'hard', 'medium', 'easy']
        }

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
        
        self.sql_generator_step1 = SQLGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            generate_num=10
        )

        self.sql_execution_filter_step2 = SQLExecutionFilter(
            database_manager=database_manager,
            timeout=sql_execution_timeout
        )

        self.text2sql_question_generator_step3 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=embedding_serving,
            database_manager=database_manager,
            question_candidates_num=5
        )

        self.text2sql_prompt_generator_step4 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=prompt_template,
            schema_config=schema_config
        )

        self.sql_cot_generator_step5 = Text2SQLCoTGenerator(
            llm_serving=cot_generation_api_llm_serving,
            database_manager=database_manager,
            schema_config=schema_config,
            max_retries=3,
            enable_retry=True,
            timeout=sql_execution_timeout
        )

        self.sql_component_classifier_step6 = SQLComponentClassifier(
            difficulty_config=component_difficulty_config
        )

        self.sql_execution_classifier_step7 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            difficulty_config=execution_difficulty_config,
            num_generations=execution_difficulty_config["num_generations"],
            timeout=sql_execution_timeout
        )
        
        
    def forward(self):

        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"

        self.sql_generator_step1.run(
            storage=self.storage.step(),
            output_sql_key=sql_key,
            output_db_id_key=db_id_key
        )

        self.sql_execution_filter_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.text2sql_question_generator_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key
        )

        self.text2sql_prompt_generator_step4.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            output_prompt_key="prompt"
        )

        self.sql_cot_generator_step5.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            output_cot_key="cot_reasoning"
        )

        self.sql_component_classifier_step6.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            output_difficulty_key="sql_component_difficulty"
        )

        self.sql_execution_classifier_step7.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_prompt_key="prompt",
            output_difficulty_key="sql_execution_difficulty"
        )

if __name__ == "__main__":
    model = Text2SQLGeneration_APIPipeline()
    model.forward()

