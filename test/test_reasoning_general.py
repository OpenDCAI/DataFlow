from dataflow.operators.generate import (
    QuestionGenerator,
    AnswerGenerator,
)
from dataflow.operators.filter import QuestionFilter, AnswerNgramFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC

"""
if the 'prompt_template' is not None and the 'content_type is set to 'diy', please check the input and output format, the same as default prompt
"""

class GeneralReasoningPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
        self.content_type = "general"
        
        self.storage = FileStorage(
            first_entry_file_name="../dataflow/example/ReasoningPipeline/pipeline_general.json",
            cache_path="./cache_local",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )

        # use API server as LLM serving
        llm_serving = APILLMServing_request(
                api_url="http://123.129.219.111:3000/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=30
        )

        self.question_filter_step1 = QuestionFilter(
            system_prompt="You are an expert in evaluating mathematical problems. Follow the user's instructions strictly and output your final judgment in the required JSON format.",
            llm_serving=llm_serving,
            content_type=self.content_type,
        #     prompt_template="""Please only keep the medical related data (judgement_test is true), for other data the judgement_test is false.
        # After these steps, output exactly:
        # {{
        #     "judgement_test": true/false,
        #     "error_type": "<error description or null>"
        # }}
        # You may include your chain of thought, but the final output must be the JSON above.

        # Here is the content to evaluate:
        # -------------------------------
        # {question}
        # -------------------------------
        # """
        )
        
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=1,
            llm_serving=llm_serving,
            content_type=self.content_type,
        #     prompt_template=""" 
        # Please construct some new sports related data from source problem.
        # Here is the problem from the user:
        # {question}
        # Write another problem inspired by this one.
        # Not only change the problem scenario, but also try to create a new problem that requires another approach to solve.
        # Start directly with the problem statement and DO NOT include any phrases such as "Here is a new problem inspired by a given one".
        # After the problem is generated finish your response right away.
        # """
        )
        
        self.answer_generator_step3 = AnswerGenerator(
            llm_serving=llm_serving,
            content_type=self.content_type,
            # prompt_template="""Please firstly output a symbol "Yeah, It is the answer:", and then output the answer."""
        )
        
        self.answer_ngram_filter_step4 = AnswerNgramFilter(
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
            question_key = "instruction",
            answer_key = "generated_cot"
        )

if __name__ == "__main__":
    pl = GeneralReasoningPipeline()
    pl.forward()
