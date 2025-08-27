from dataflow.operators.reasoning import (
    QuestionGenerator,
    AnswerGenerator,
)
from dataflow.operators.reasoning import QuestionFilter, AnswerNgramFilter, AnswerModelJudge
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request, LocalModelLLMServing
from dataflow.core import LLMServingABC
from dataflow.prompts.reasoning.general import (
    GeneralQuestionFilterPrompt,
    GeneralAnswerGeneratorPrompt,
    GeneralQuestionSynthesisPrompt,
    AnswerJudgePrompt,
)

class GeneralReasoningPipeline():
    def __init__(self, llm_serving: LLMServingABC = None):
        
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
            prompt_template=GeneralQuestionFilterPrompt()
        )
        
        self.question_gen_step2 =  QuestionGenerator(
            num_prompts=1,
            llm_serving=llm_serving,
            prompt_template=GeneralQuestionSynthesisPrompt()
        )
        
        self.answer_generator_step3 = AnswerGenerator(
            llm_serving=llm_serving,
            prompt_template=GeneralAnswerGeneratorPrompt()
        )
        self.answer_model_judge_step4 = AnswerModelJudge(
            llm_serving=llm_serving,
            prompt_template=AnswerJudgePrompt(),
            keep_all_samples=True
        )
        self.answer_ngram_filter_step5 = AnswerNgramFilter(
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
        ),
        self.answer_model_judge_step4.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "generated_cot",
            input_reference_key = "golden_answer"
        ),
        self.answer_ngram_filter_step5.run(
            storage = self.storage.step(),
            input_question_key = "instruction",
            input_answer_key = "generated_cot"
        )

if __name__ == "__main__":
    pl = GeneralReasoningPipeline()
    pl.forward()
