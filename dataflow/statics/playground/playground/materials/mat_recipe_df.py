from dataflow.operators.core_text import PromptedGenerator
from dataflow.operators.chemistry import EvaluateSmilesEquivalence

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.prompts.chemistry import ExtractSmilesFromTextPrompt

prompt = """Extract the polyimide recipe information from the text and format it as a structured JSON object.
    Follow these rules strictly:
    Only use the name in "meta-meta-linked BPDA-ODA PAE", "meta-para-linked BPDA-ODA PAE", "para-para-linked BPDA-ODA PAE", "mixed BPDA-ODA PAE"
    1. For name:
       - Use the sample name if provided, otherwise generate a name based on the dianhydride and diamine
       - Example: "PI-1" or "PI-2"

    2. For dianhydride and diamine:
       - Use ComponentInfo structure with components array and ratio array
       - All ratios are molar ratios
       - List all monomers in components array
       - List corresponding molar ratios in ratio array
       - Use abbreviation of the monomer name if possible
       - Example for dianhydride: {{
           "components": ["4-ODA", "6-FDA"],
           "ratio": [1, 1]
         }}
       - Example for single monomer: {{
           "components": ["6-FDA"],
           "ratio": [1]
         }}

    3. General rules:
  
       - Extract ALL recipes mentioned in the text
       - Each recipe should have a unique name
       - If multiple recipes are mentioned, return them as a list in the "recipes" field
       - Do not include any polyimide name from references
       - If the text is not related to polyimide, return empty string

    Example input texts and their corresponding outputs:

    1. Simple case:
    Input: "The polyimide PI-1 was synthesized from 4-ODA and 6-FDA (1:1) with HPMDA and 3,4'-ODA (1:1)."
    Output:  
        [
            {{
                "name": "PI-1",
                "dianhydride": {{
                    "components": ["4-ODA", "6-FDA"],
                    "ratio": [1, 1]
                }},
                "diamine": {{
                    "components": ["HPMDA", "3,4'-ODA"],
                    "ratio": [1, 1]
                }}
            }}
        ]

    2. Multiple recipes case:
    Input: "Two polyimides were prepared: (1) PI-1 from 6-FDA and ODA (1:1), and (2) PI-2 from PMDA and PDA (1:1)."
    Output:
    [
            {{
                "name": "PI-1",
                "dianhydride": {{
                    "components": ["6-FDA"],
                    "ratio": [1]
                }},
                "diamine": {{
                    "components": ["ODA"],
                    "ratio": [1]
                }}
            }},
            {{
                "name": "PI-2",
                "dianhydride": {{
                    "components": ["PMDA"],
                    "ratio": [1]
                }},
                "diamine": {{
                    "components": ["PDA"],
                    "ratio": [1]
                }}
            }}
        ]
    """

class ExtractSmiles():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/chemistry/matched_sample_10.json",
            #first_entry_file_name="/Users/lianghao/Desktop/dataflow_code/test_dataflow/test/matched_sample_10.json",
            cache_path="./cache_all_3",
            file_name_prefix="math_QA",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gemini-2.5-flash",
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
