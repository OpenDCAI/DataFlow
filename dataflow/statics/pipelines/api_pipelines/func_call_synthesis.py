from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.operators.conversations import (
    ScenarioExtractor,
    ScenarioExpander,
    AtomTaskGenerator,
    SequentialTaskGenerator,
    ParaSeqTaskGenerator,
    CompositionTaskFilter,
    FunctionGenerator,
    MultiTurnConversationGenerator
)

class FuncCall_APIPipeline:
    def __init__(self):

        self.storage = FileStorage(
            first_entry_file_name="../example_data/FuncCallPipeline/chat_data.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
      
        self.llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=128
        )
        
        self.scenario_extractor = ScenarioExtractor(llm_serving=self.llm_serving)
        self.scenario_expander = ScenarioExpander(llm_serving=self.llm_serving)
        self.atom_task_generator = AtomTaskGenerator(llm_serving=self.llm_serving)
        self.sequential_task_generator = SequentialTaskGenerator(llm_serving=self.llm_serving)
        self.parallel_sequential_stak_generator = ParaSeqTaskGenerator(llm_serving=self.llm_serving)
        self.composition_task_filter = CompositionTaskFilter(llm_serving=self.llm_serving)
        self.function_generator = FunctionGenerator(llm_serving=self.llm_serving)
        self.multi_turn_conversations_generator = MultiTurnConversationGenerator(llm_serving=self.llm_serving)

    def run(self):
       self.scenario_extractor.run(
           self.storage.step(),
           input_chat_key="chat"
       )
       self.scenario_expander.run(
           self.storage.step(),
           input_scenario_key="scenario"
       )
       self.atom_task_generator.run(
           self.storage.step(),
           input_scenario_key="scenario"
       )
    #    self.atom_task_generator.run(
    #        self.storage.step(),
    #        input_scenario_key="modified_scenario",
    #        output_key='subsequent_task'
    #    )
       self.sequential_task_generator.run(
           self.storage.step(),
           input_task_key="atom_task"
       )
    #    self.parallel_sequential_stak_generator.run(
    #        self.storage.step(),
    #        input_task_key="atom_task"
    #    )
       self.composition_task_filter.run(
           self.storage.step(),
           input_composition_task_key="composition_task",
           input_sub_tasks_keys=["atom_task", "subsequent_task"]
       )
       self.function_generator.run(
           self.storage.step(),
           input_composition_task_key="composition_task",
           input_sub_tasks_keys=["atom_task", "subsequent_task"]
       )
       self.multi_turn_conversations_generator.run(
           self.storage.step(),
           input_task_key="composition_task",
           input_sub_tasks_keys=["atom_task", "subsequent_task"],
           input_functions_key="functions",
        )
    

if __name__ == "__main__":
    pipeline = FuncCall_APIPipeline()
    pipeline.run()