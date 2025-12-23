 
from dataflow.operators.text_sft import CondorGenerator
from dataflow.operators.core_text import PromptedGenerator,PromptTemplatedGenerator
from dataflow.operators.core_text import GeneralFilter
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.prompts.core_text import StrFormatPrompt

prompt = (
        "You are a senior content quality evaluator performing final assessment. "
        "Rate the processed content on a 1-5 scale for overall excellence and utility.\n\n"
        "Final Scoring Criteria:\n"
        "• 1: Poor - significant quality issues remain\n"
        "• 2: Below standard - limited value or clarity problems\n"
        "• 3: Acceptable - meets basic requirements but unremarkable\n"
        "• 4: High quality - well-crafted, informative, and valuable\n"
        "• 5: Excellent - exceptional content that serves as exemplary reference\n\n"
        "Evaluate holistically:\n"
        "- Information completeness and accuracy\n"
        "- Clarity and readability\n"
        "- Practical value and usefulness\n"
        "- Structure and logical flow\n"
        "- Overall contribution to knowledge/understanding\n\n"
        "Only content scoring 4-5 will be retained for final dataset. "
        "Be appropriately selective to ensure high standards.\n\n"
        "Respond with only the integer score (1-5)."
    )
class SFT_From_Scratch():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/GeneralTextPipeline/empty.jsonl",
            cache_path="./cache",
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl",
        )
        self.model_cache_dir = './dataflow_cache_1'
        self.num_generated_samples = 3
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=100
        )
        self.instruction_generator = CondorGenerator(llm_serving=self.llm_serving, num_samples=self.num_generated_samples)
        self.answer_generator = PromptedGenerator(llm_serving=self.llm_serving, system_prompt="Please answer this question.")
        self.prompt_template = StrFormatPrompt(
            f_str_template="Please rate the following SFT data: instruction: {instruction}, output: {output}?"
        )
        self.rater = PromptTemplatedGenerator(llm_serving=self.llm_serving, system_prompt = prompt, prompt_template=self.prompt_template)
        self.filter = GeneralFilter([lambda df: df['data_score'] >= 3])

    def forward(self):
        self.instruction_generator.run(
            storage=self.storage.step()
        )
        self.answer_generator.run(
            storage=self.storage.step(),
            input_key='instruction',
            output_key="output",
        )
        self.rater.run(
            storage=self.storage.step(),
            output_key = "data_score",
            instruction = "instruction",
            output = "output",
        )
        self.filter.run(
            storage=self.storage.step(),
        )

if __name__ == "__main__":
    model = SFT_From_Scratch()
    model.forward()
