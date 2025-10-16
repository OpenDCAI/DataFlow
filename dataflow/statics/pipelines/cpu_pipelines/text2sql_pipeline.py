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
    local_dir = "./hf_cache"
    extract_to = "./downloaded_databases"
    
    logger.info(f"Downloading and extracting database from {dataset_repo_id}...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)
    
    downloaded_path = snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True
    )
    
    logger.info(f"Files downloaded to: {downloaded_path}")
    
    zip_path = os.path.join(downloaded_path, "databases.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Database files extracted to {extract_to}")
        return extract_to
    else:
        raise FileNotFoundError(f"Database zip file not found at {zip_path}")


class Text2SQL_CPUPipeline():
    def __init__(self, db_root_path=""):
        self.logger = get_logger()
        self.db_root_path = db_root_path

        if not db_root_path:
            try:
                self.db_root_path = download_and_extract_database(self.logger)
                self.logger.info(f"Using automatically downloaded database at: {self.db_root_path}")
            except Exception as e:
                self.logger.error(f"Failed to auto-download database: {e}")
                raise 
        else:
            self.logger.info(f"Using manually specified database path: {self.db_root_path}")

        if not os.path.exists(self.db_root_path):
            raise FileNotFoundError(f"Database path does not exist: {self.db_root_path}")

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
    # If you have your own database files, you can set the db_root_path to the path of your database files
    # If not, please set the db_root_path "", and we will download the example database files automatically
    db_root_path = ""

    model = Text2SQL_CPUPipeline(db_root_path=db_root_path)
    model.forward()

