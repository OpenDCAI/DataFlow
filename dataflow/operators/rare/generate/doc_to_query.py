import pandas as pd
import json
from dataflow.utils.registry import OPERATOR_REGISTRY # Changed from GENERATOR_REGISTRY to OPERATOR_REGISTRY
from dataflow import get_logger # Simplified import
from dataflow.utils.storage import DataFlowStorage # New import for storage
from dataflow.core import OperatorABC # New import for OperatorABC
from dataflow.core import LLMServingABC # New import for LLMServingABC

DOC2HARD_QUERY = '''# Context
You are tasked with generating reasoning-intensive questions with scenarios based on a given document. These questions must be standalone (meaningful without the document) while being answerable using information from the document as supporting evidence. The questions should specifically engage with core concepts and principles from the document's domain.

# Question Requirements
1. Each question MUST:
- Present a complete scenario or context within itself
- Be answerable through logical reasoning and critical thinking
- Remain valid and meaningful even if the source document didn't exist
- Target higher-order thinking skills (analysis, evaluation, synthesis)
- Be domain-relevant but not document-specific
- Incorporate key concepts, terminology, and principles from the document's field
- Challenge understanding of domain-specific problem-solving approaches

2. Each question MUST NOT:
- Directly reference the document or its contents
- Be answerable through simple fact recall
- Require specific knowledge only found in the document
- Be a reading comprehension question
- Stray from the core subject matter of the document's domain

# Domain Alignment Guidelines
Before generating questions:
1. Identify the primary domain (e.g., programming, medicine, economics)
2. Extract key concepts and principles from the document
3. List common problem-solving patterns in this domain

When crafting questions:
1. Frame scenarios using domain-specific contexts
2. Incorporate relevant technical terminology naturally
3. Focus on problem-solving approaches typical to the field
4. Connect theoretical concepts to practical applications within the domain

After generating questions step by step, reformat questions including corresponding scenarios in JSON with key "hard_query":
```json
{{
    "hard_query": { "question": <str>, "scenario": <str>}
}}
```
Now, ** the number of hard_queries to generate is exactly 1 **.

# Document
'''

@OPERATOR_REGISTRY.register() # Changed decorator to OPERATOR_REGISTRY
class Doc2Query(OperatorABC): # Inherit from OperatorABC
    '''
    Doc2Query uses LLMs to generate reasoning-intensive questions for given documents.
    '''

    def __init__(self, llm_serving: LLMServingABC): # Changed config to llm_serving
        self.logger = get_logger()
        self.llm_serving = llm_serving # Renamed generator to llm_serving

        # Removed config related attributes as they will be passed in run method
        self.input_key = "text"
        self.output_question_key = "question"
        self.output_scenario_key = "scenario"
        self.max_attempts = 3 # Default value, can be made configurable in run method if needed

    @staticmethod
    def get_desc(lang: str = "zh"): # Removed self as it's a static method
        if lang == "zh":
            return (
                "RAREPipeline: Doc2Query 算子使用大语言模型为给定文档生成推理密集型问题。\n\n"
                "输入参数：\n"
                "- input_key: 包含文档片段的字段名\n"
                "- output_question_key: 包含生成问题的字段名\n"
                "- output_scenario_key: 包含生成情景的字段名\n"
            )
        elif lang == "en":
            return (
                "RAREPipeline: Doc2Query operator uses LLMs to generate reasoning-intensive questions for given documents.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the content\n"
                "- output_question_key: Field name containing the generated question\n"
                "- output_scenario_key: Field name containing the generated scenario\n"
            )
        else:
            return "RAREPipeline: Doc2Query operator uses LLMs to generate reasoning-intensive questions for given documents."

    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = [self.output_question_key, self.output_scenario_key]

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            raise ValueError(f"Missing required column(s): {missing}")
        if conflict:
            raise ValueError(f"The following column(s) already exist and would be overwritten: {conflict}")

    def _build_prompt(self, df) -> list: # Changed return type hint to list
        prompts = []
        for index, row in df.iterrows():
            prompt = DOC2HARD_QUERY + row[self.input_key]
            prompts.append(prompt)
        return prompts # Return as a list directly

    def run(
        self,
        storage: DataFlowStorage,
        input_key: str = "text",
        output_question_key: str = "question",
        output_scenario_key: str = "scenario",
        max_attempts: int = 3 # Added max_attempts as a parameter
    ):
        '''
        Runs the reasoning-intensive question generation process, reading from the input storage and saving results to output.
        '''
        self.input_key = input_key
        self.output_question_key = output_question_key
        self.output_scenario_key = output_scenario_key
        self.max_attempts = max_attempts

        dataframe = storage.read("dataframe") # Read from storage
        self._validate_dataframe(dataframe)
        prompts = self._build_prompt(dataframe)
        responses = self.llm_serving.generate_from_input(user_inputs=prompts, system_prompt="") # Updated LLM call

        questions, scenarios = [], []
        for idx, response in enumerate(responses):
            attempts = 0
            while attempts < self.max_attempts:
                try:
                    response = response.strip()
                    decoder = json.JSONDecoder()
                    start_idx = response.find('{')
                    if start_idx == -1:
                        self.logger.error(f"Invalid response format: {response}")
                        raise ValueError("Response does not contain a valid JSON object.")
                    json_data, _ = decoder.raw_decode(response[start_idx:])
                    questions.append(json_data['hard_query']['question'])
                    scenarios.append(json_data['hard_query']['scenario'])
                    break
                except Exception as e:
                    self.logger.error(f"Error parsing response: {response}, error: {e}")
                    attempts += 1
                    # Re-generate if parsing fails
                    response = self.llm_serving.generate_from_input(user_inputs=[prompts[idx]], system_prompt="")[0].strip()
                    if attempts >= self.max_attempts:
                        questions.append("")
                        scenarios.append("")

        dataframe[self.output_question_key] = questions
        dataframe[self.output_scenario_key] = scenarios

        output_file = storage.write(dataframe) # Write to storage
        self.logger.info(f"Results saved to {output_file}")

        return [self.output_question_key, self.output_scenario_key]

