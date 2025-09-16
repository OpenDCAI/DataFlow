from dataflow.operators.reasoning import (
    ReasoningQuestionGenerator,
    ReasoningAnswerGenerator,
)
from dataflow.operators.reasoning import ReasoningQuestionFilter, ReasoningAnswerNgramFilter

from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.diy import (
    DiyQuestionFilterPrompt,
    DiyAnswerGeneratorPrompt,
    DiyQuestionSynthesisPrompt
)

"""
if the 'prompt_template' is not None and the 'content_type is set to 'diy', please check the input and output format, the same as default prompt
"""

DIY_PROMPT_QUESTION = """Please only keep the medical related data (judgement_test is true), for other data the judgement_test is false.
        After these steps, output exactly:
        {{
            "judgement_test": true/false,
            "error_type": "<error description or null>"
        }}
        You may include your chain of thought, but the final output must be the JSON above.

        Here is the content to evaluate:
        -------------------------------
        {question}
        -------------------------------
        """
    
DIY_PROMPT_SYNTHESIS = """
    Please construct some new sports related data from source problem.
    Here is the problem from the user:
    {question}
    Write another problem inspired by this one.
    Not only change the problem scenario, but also try to create a new problem that requires another approach to solve.
    Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
    After the problem is generated finish your response right away.
    """
    
DIY_PROMPT_ANSWER = """Please firstly output a symbol "Yeah, It is the answer:", and then output the answer."""

class DiyReasoning_APIPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
        self.storage = FileStorage(
            first_entry_file_name="../example_data/ReasoningPipeline/pipeline_general.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        self.llm_serving = APILLMServing_request(
                    api_url="http://api.openai.com/v1/chat/completions",
                    model_name="gpt-4o",
                    max_workers=30
        )

        self.question_filter_step1 = ReasoningQuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=self.llm_serving,
            prompt_template=DiyQuestionFilterPrompt(DIY_PROMPT_QUESTION)
        )
        
        self.question_gen_step2 =  ReasoningQuestionGenerator(
            num_prompts=1,
            llm_serving=self.llm_serving,
            prompt_template=DiyQuestionSynthesisPrompt(DIY_PROMPT_SYNTHESIS)
        )
        
        self.answer_generator_step3 = ReasoningAnswerGenerator(
            llm_serving=self.llm_serving,
            prompt_template=DiyAnswerGeneratorPrompt(DIY_PROMPT_ANSWER)
        )
        
        self.answer_ngram_filter_step4 = ReasoningAnswerNgramFilter(
            min_score = 0.1,
            max_score = 1.0,
            ngrams = 5
        )
        
    def forward(self):
        self.question_filter_step1.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )

        self.question_gen_step2.run(
            storage = self.storage.step(),
            input_key = "instruction",
        )
        self.answer_generator_step3.run(
            storage = self.storage.step(),
            input_key = "instruction", 
            output_key = "generated_cot"
        )
        self.answer_ngram_filter_step4.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "generated_cot"
        )

if __name__ == "__main__":
    pl = DiyReasoning_APIPipeline()
    pl.forward()
