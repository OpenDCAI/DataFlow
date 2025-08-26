import os
import zipfile
from dataflow import get_logger
from pathlib import Path
from huggingface_hub import snapshot_download

from dataflow.operators.text2sql import (
    Text2SQLPromptGenerator
)
from dataflow.operators.text2sql import (
    SQLExecutionFilter
)
from dataflow.operators.text2sql import (
    SQLComponentClassifier
)
from dataflow.prompts.text2sql import (
    Text2SQLPromptGeneratorPrompt
)
from dataflow.utils.storage import FileStorage
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


class Text2SQL_CPUPipeline():
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
            first_entry_file_name="../example_data/Text2SQLPipeline/pipeline_refine.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
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
            sql_execution_timeout = 2,
            max_connections_per_db=100,
            max_workers=100
        )

        self.sql_execution_filter_step1 = SQLExecutionFilter(
            database_manager=database_manager,
        )

        self.text2sql_prompt_generator_step2 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt()
        )

        self.sql_component_classifier_step3 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=['easy', 'medium', 'hard', 'extra']
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
    model = Text2SQL_CPUPipeline(auto_download_db=False)
    model.forward()

