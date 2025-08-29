from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.chemistry import EvaluateSmilesEquivalence

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.prompts.chemistry import ExtractSmilesFromTextPrompt

base_prompt = """You are an expert in polymer materials.
    Several polymers are mentioned in this article.
    Your task is to extract the following glass transition temperature (Tg)–related information for each polymer as explicitly stated in the article only (do not use general knowledge or cited references).
    If any information is not present in the article, leave the corresponding cell empty.

    Field descriptions: 
    - name: material name.
    - tg: The glass transition temperature of the polymer, including its unit (e.g. "250 °C").
    - method: The method used to measure Tg (e.g. DSC, DMA).
    - parameter: Experimental parameters of the Tg measurement, such as scan cycle, scan range, heating rate, frequency, etc. Do not include parameters unrelated to Tg measurements (e.g. cure process parameters).
    - instrument: The model of the device used to measure Tg.
    - tg_note: Supplementary information for the Tg data, such as the state of the material, synthesis process, test process, cycle number, or other remarks.

    Output format:
    Return the result as a raw CSV table with the header: name,tg,method,parameter,instrument,tg_note

    Requirments:
    - List each polymer as one separate row under the header.
    - If multiple variants or versions of the same polymer are tested under different conditions, list all combinations as separate rows (e.g. Cartesian product if applicable).
    - Do not output JSON, Markdown, or any extra explanatory text.
    - Do not add any text before or after the CSV table.
    - For each material, if there is no Tg data, do not output the material.

    Example output:
    name,tg,method,parameter,instrument,tg_note
    Polymer A,250 °C,DSC,5 °C/min to 400 °C,TA Q200,as-cured
    Polymer B,180 °C,DMA,1 Hz; 3 °C/min,TA Q800,cured 2 h at 180 °C
    """

system_prompt = base_prompt + f"\nThis rule has the highest priority: Only extract information for the following materials:\n \"meta-meta-linked BPDA-ODA PAE\", \"meta-para-linked BPDA-ODA PAE\", \"para-para-linked BPDA-ODA PAE\", \"mixed BPDA-ODA PAE\" "

class ExtractSmiles():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/chemistry/matched_sample_10.json",
            #first_entry_file_name="/Users/lianghao/Desktop/dataflow_code/test_dataflow/test/matched_sample_10.json",
            cache_path="./cache_all_2",
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
            system_prompt=system_prompt
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
