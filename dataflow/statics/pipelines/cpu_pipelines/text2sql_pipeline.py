from dataflow.operators.generate import *
from dataflow.utils.storage import FileStorage


class Text2SQLPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/Text2SQLPipeline/pipeline.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # A demo database is provided. Download it from the following URL and update the path:  
        # https://huggingface.co/datasets/Open-Dataflow/dataflow-Text2SQL-database-example  
        db_root_path = ""  
        table_info_file = "../example_data/Text2SQLPipeline/dev_tables.jsonl"

        self.sql_difficulty_classifier_step1 = SQLDifficultyClassifier()

        self.schema_linking_step2 = SchemaLinking(
            table_info_file=table_info_file
        )

        self.database_schema_extractor_step3 = DatabaseSchemaExtractor(
            table_info_file=table_info_file,
            db_root_path=db_root_path,
        )
        
        
    def forward(self):

        input_sql_key = "SQL"
        input_dbid_key = "db_id"

        self.sql_difficulty_classifier_step1.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            output_difficulty_key="sql_component_difficulty"
        )

        self.schema_linking_step2.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_dbid_key=input_dbid_key,
            output_used_schema_key="selected_schema"        
        )

        self.database_schema_extractor_step3.run(
            storage=self.storage.step(),
            input_db_key=input_dbid_key,
            table_schema_file_db_key="db_id",
            output_raw_schema_key="whole_schema",
            output_ddl_key="ddl",
            output_whole_format_schema_key="whole_format_schema"
        )
        
if __name__ == "__main__":
    model = Text2SQLPipeline()
    model.forward()

