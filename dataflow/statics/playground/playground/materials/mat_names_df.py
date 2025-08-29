from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.chemistry import EvaluateSmilesEquivalence

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.prompts.chemistry import ExtractSmilesFromTextPrompt

prompt = """You are an expert in materials science. You need analyze the composition and synthesis/process of the materials and extract the material names from the text.
    Follow these rules strictly:
    1. For composition and synthesis process analysis:
        - Describe all materials and their contents, synthesis process, etc. you find in the text, like:
         Base materials: Material1, Material2, Material3
         Modifiers identified: Additive with contents of 1 mass%, 3 mass%, and 5 mass%
         Additional components: Curing agent, etc.
         Synthesis: The materials were synthesized by mixing the base materials and modifiers.
         Process: The materials were annealed at 1000°C for 1 hour.

    2. For material names:
        -  List all materials that were actually tested, following these rules strictly, only use to distinguish different materials:
         - Include only materials with measured properties, this rule has priority over the other rules.
         - Use the abbreviations of sample provided by the author if possible, such as P1, P2, P3, etc.
         - If no name is provided, generate a unique name based on the composition and synthesis process. The process is not required, and only needed when different materials with different processes are tested.
         - The name should be as concise as possible, and should not be too long.
         - If the text is not related to materials, return empty string.

    3. For material types:
        - List the material type corresponding to each material listed above in material names.
        - Use "linear" or "cross_linked" to indicate the structure of polymers, instead of "polymer".
        - If the polymer structure ("linear" or "cross_linked") is not specified, default to "cross_linked".
        - Ensure the number of material types matches the number of material names.
    Do not include explanations, markdown formatting, or code fences (```).
    """

class ExtractSmiles():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/chemistry/matched_sample_10.json",
            #first_entry_file_name="/Users/lianghao/Desktop/dataflow_code/test_dataflow/test/matched_sample_10.json",
            cache_path="./cache_all_1",
            file_name_prefix="math_QA",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gpt-4o",
                max_workers=200,
                response_format = ""
        )
        self.prompt_generator = PromptedGenerator(
            llm_serving = self.llm_serving, 
            system_prompt=prompt
        )
        self.smile_eval = EvaluateSmilesEquivalence()

    def forward(self):
        # Initial filters
        self.prompt_generator.run(
            storage = self.storage.step(),
            input_key = "text",
            output_key = "synth"
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = ExtractSmiles()
    model.forward()
