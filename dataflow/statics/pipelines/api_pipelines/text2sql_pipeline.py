from dataflow.operators.generate import *
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request


class Text2SQLPipeline():
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/Text2SQLPipeline/pipeline.json",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        api_llm_serving = APILLMServing_request(
            api_url="https://api.openai.com/v1/chat/completions",
            model_name="gpt-4o",
            max_workers=100
        )

        # A demo database is provided. Download it from the following URL and update the path:  
        # https://huggingface.co/datasets/Open-Dataflow/dataflow-Text2SQL-database-example  
        db_root_path = ""  
        table_info_file = "../example_data/Text2SQLPipeline/dev_tables.jsonl"
        
        self.sql_filter_step1 = SQLFilter(
            llm_serving=api_llm_serving,
            db_root_path=db_root_path,
            num_cpus=20,
            meta_time_out=120
        )

        self.sql_difficulty_classifier_step2 = SQLDifficultyClassifier()

        self.schema_linking_step3 = SchemaLinking(
            table_info_file=table_info_file
        )

        self.database_schema_extractor_step4 = DatabaseSchemaExtractor(
            table_info_file=table_info_file,
            db_root_path=db_root_path,
        )

        self.extra_knowledge_generator_step5 = ExtraKnowledgeGenerator(
            llm_serving=api_llm_serving,
            exist_knowledge=False,
            max_retries=2,
            batch_size=50
        )

        self.question_refiner_step6 = QuestionRefiner(
            llm_serving=api_llm_serving,
            num_threads=5,
            max_retries=3
        )

        self.prompt_generator_step7 = PromptGenerator(
            llm_serving=api_llm_serving,
            db_root_path=db_root_path,
            num_threads=5,
            timeout=60
        )

        self.text2sql_difficulty_classifier_step8 = Text2SQLDifficultyClassifier(
            llm_serving=api_llm_serving,
            db_root_path=db_root_path,
            num_cpus=1, 
            meta_time_out=120.0
        )
        
        
    def forward(self):

        input_sql_key = "SQL"
        input_dbid_key = "db_id"
        input_question_key = "question"

        self.sql_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_dbid_key=input_dbid_key,
            input_question_key=input_question_key
        )

        self.sql_difficulty_classifier_step2.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            output_difficulty_key="sql_component_difficulty"
        )

        self.schema_linking_step3.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_dbid_key=input_dbid_key,
            output_used_schema_key="selected_schema"        
        )

        self.database_schema_extractor_step4.run(
            storage=self.storage.step(),
            input_db_key=input_dbid_key,
            table_schema_file_db_key="db_id",
            output_raw_schema_key="whole_schema",
            output_ddl_key="ddl",
            output_whole_format_schema_key="whole_format_schema"
        )

        self.extra_knowledge_generator_step5.run(
            storage=self.storage.step(),
            input_question_key=input_question_key,
            input_sql_key=input_sql_key,
            input_schema_key="ddl",
            output_knowledge_key="evidence"
        )

        self.question_refiner_step6.run(
            storage=self.storage.step(),
            input_question_key=input_question_key,
            output_refined_question_key="refined_question"
        )

        self.prompt_generator_step7.run(
            storage=self.storage.step(),
            input_sql_key=input_sql_key,
            input_question_key=input_question_key,
            input_dbid_key=input_dbid_key,
            input_schema_key="ddl",
            output_sft_prompt_key="sft_prompt",
            output_rl_prompt_key="rl_prompt",
            output_cot_key="sft_output"
        )

        self.text2sql_difficulty_classifier_step8.run(
            storage=self.storage.step(),
            input_dbid_key=input_dbid_key,
            input_sql_key=input_sql_key,
            input_prompt_key="rl_prompt",
            output_difficulty_key="sql_execution_difficulty"
        )
        
if __name__ == "__main__":
    model = Text2SQLPipeline()
    model.forward()

