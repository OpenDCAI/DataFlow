import os
from dataflow import get_logger
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

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
from dataflow.prompts.text2sql import (
    Text2SQLCotGeneratorPrompt,
    SelectSQLGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import LocalModelLLMServing_vllm, LocalModelLLMServing_sglang
from dataflow.utils.text2sql.database_manager import DatabaseManager

def download_and_extract_database(logger):
    dataset_repo_id = "Open-Dataflow/dataflow-Text2SQL-database-example"
    subfolder = "databases"
    local_dir = "./hf_cache"
    extract_to = "./downloaded_databases"
    logger.info(f"Downloading and extracting database from {dataset_repo_id}...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        allow_patterns=f"{subfolder}/*",  
        local_dir=local_dir,
        resume_download=True  
    )
    logger.info(f"Database files downloaded to {local_dir}/{subfolder}")

    zip_path = Path(local_dir) / subfolder / "databases.zip"
    extract_path = Path(extract_to)

    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        logger.info(f"Database files extracted to {extract_path}")
        return str(extract_path) 
    else:
        raise FileNotFoundError(f"Database files not found in {zip_path}")

class Text2SQLGeneration_GPUPipeline():
    def __init__(self, auto_download_db=False):

        self.logger = get_logger()
        self.db_root_path = "" 

        if auto_download_db:
            try:
                self.db_root_path = download_and_extract_database(self.logger)
                self.logger.info(f"Using automatically downloaded database at: {self.db_root_path}")
            except Exception as e:
                self.logger.error(f"Failed to auto-download database: {e}")
                raise 
        else:
             if not self.db_root_path:
                self.logger.error(
                    "Auto-download is disabled and 'db_root_path' is not set. "
                    "Please manually assign the path to the database files to 'self.db_root_path' "
                    "before initializing the DatabaseManager, or set auto_download_db=True."
                )
                raise ValueError("Database path is not specified, please specify the database path manually.")
             else:
                 self.logger.info(f"Using manually specified database path: {self.db_root_path}")

        self.storage = FileStorage(
            first_entry_file_name="",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        self.llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
        )

        # use SGLang as LLM serving
        # llm_serving = LocalModelLLMServing_sglang(
        #     hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
        #     sgl_dp_size=1, # data parallel size
        #     sgl_tp_size=1, # tensor parallel size
        #     sgl_max_tokens=1024,
        #     sgl_tensor_parallel_size=4
        # )

        # It is recommended to use better LLMs for the generation of Chain-of-Thought (CoT) reasoning process.
        cot_generation_llm_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Qwen/Qwen2.5-7B-Instruct", # set to your own model path
            vllm_tensor_parallel_size=1,
            vllm_max_tokens=8192,
        )

        embedding_serving = LocalModelLLMServing_vllm(
            hf_model_name_or_path="Alibaba-NLP/gte-Qwen2-7B-instruct", 
            vllm_max_tokens=8192
        )

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
                "root_path": self.db_root_path
            },
            logger=None,
            max_connections_per_db=100,
            max_workers=100
        )
        
        self.sql_generator_step1 = SQLGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            generate_num=10,
            prompt_template=SelectSQLGeneratorPrompt()
        )

        self.sql_execution_filter_step2 = SQLExecutionFilter(
            database_manager=database_manager,
        )

        self.text2sql_question_generator_step3 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=embedding_serving,
            database_manager=database_manager,
            question_candidates_num=5,
            prompt_template=Text2SQLQuestionGeneratorPrompt()
        )

        self.text2sql_prompt_generator_step4 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt()
        )

        self.sql_cot_generator_step5 = Text2SQLCoTGenerator(
            llm_serving=cot_generation_api_llm_serving,
            database_manager=database_manager,
            max_retries=3,
            enable_retry=True,
            prompt_template=Text2SQLCotGeneratorPrompt()
        )

        self.sql_component_classifier_step6 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=['easy', 'medium', 'hard', 'extra']
        )

        self.sql_execution_classifier_step7 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            num_generations=10,
            difficulty_thresholds=[2, 5, 9],
            difficulty_labels=['extra', 'hard', 'medium', 'easy']
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
    model = Text2SQLGeneration_GPUPipeline(auto_download_db=True)
    model.forward()