from dataflow.operators.generate import (
    SQLGenerator,
    QuestionGeneration,
    PromptGenerator,
    CoTGenerator
)
from dataflow.operators.filter import (
    ExecutionFilter
)
from dataflow.operators.eval import (
    ComponentClassifier,
    ExecutionClassifier
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing_vllm
from dataflow.utils.text2sql.database_manager import DatabaseManager


class Text2SQLPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_gen.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )

        # It is recommended to use better LLMs for the generation of Chain-of-Thought (CoT) reasoning process.
        cot_generation_api_llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o", # You can change to a more powerful model for CoT generation
            max_workers=100
        )

        embedding_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/embeddings",
            model_name="text-embedding-ada-002",
            max_workers=100
        )

        # You can customize the difficulty config here, but it must contain 'thresholds' and 'labels' keys
        execution_difficulty_config = {
            'thresholds': [2, 5, 9],
            'labels': ['easy', 'medium', 'hard', 'extra']
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
            }
        )
        
        self.sql_generator_step1 = SQLGenerator(
            llm_serving=api_llm_serving,
            database_manager=database_manager,
            generate_num=300
        )

        self.sql_execution_filter_step2 = ExecutionFilter(
            database_manager=database_manager
        )

        self.text2sql_question_generator_step3 = QuestionGeneration(
            llm_serving=api_llm_serving,
            embedding_serving=embedding_serving,
            database_manager=database_manager,
            question_candidates_num=5
        )

        self.text2sql_prompt_generator_step4 = PromptGenerator(
            database_manager=database_manager,
            prompt_template=prompt_template,
            schema_config=schema_config
        )

        self.sql_cot_generator_step5 = CoTGenerator(
            llm_serving=cot_generation_api_llm_serving,
            database_manager=database_manager,
            schema_config=schema_config,
            max_retries=3,
            enable_retry=True
        )

        self.sql_component_classifier_step6 = ComponentClassifier(
            difficulty_config=component_difficulty_config
        )

        self.sql_execution_classifier_step7 = ExecutionClassifier(
            llm_serving=api_llm_serving,
            database_manager=database_manager,
            difficulty_config=execution_difficulty_config,
            num_generations=5
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
    model = Text2SQLPipeline()
    model.forward()

