from dataflow.operators.chemistry import ExtractSmilesFromText
from dataflow.operators.chemistry import EvaluateSmilesEquivalence

from dataflow.serving import APILLMServing_request
from dataflow.utils.storage import FileStorage
from dataflow.prompts.chemistry import ExtractSmilesFromTextPrompt

smiles_prompt = """Extract the monomer/small molecule information from the text and format it as a structured JSON object.
    Follow these rules strictly:
    1. For each monomer/small molecule, extract:
       - abbreviation: The commonly used abbreviated name
       - full_name: The complete chemical name
       - smiles: The SMILES notation of the molecular structure

    2. General rules:
       - Each monomer/small molecule should have a unique abbreviation
       - If a monomer's information is incomplete, include only the available information
       - Don't recognize polymer which have "poly" in the name as monomer

    Example output:
        [
            {
                "abbreviation": "4-ODA",
                "full_name": "4,4′-Oxydianiline",
                "smiles": "O(c1ccc(N)cc1)c2ccc(cc2)N"
            },
            {
                "abbreviation": "6FDA",
                "full_name": "4,4'-(hexafluoroisopropylidene)diphthalic anhydride",
                "smiles": "C1=CC2=C(C=C1C(C3=CC4=C(C=C3)C(=O)OC4=O)(C(F)(F)F)C(F)(F)F)C(=O)OC2=O"
            }
        ]
    Please make sure to output pure json which can be saved into a json file, do not output like html.
    """

response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "chemical_structures_response",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "chemical_structures": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "abbreviation": {
                "type": "string"
              },
              "full_name": {
                "type": "string"
              },
              "smiles": {
                "type": "string"
              }
            },
            "required": ["abbreviation", "full_name", "smiles"],
            "additionalProperties": False
          }
        }
      },
      "required": ["chemical_structures"],
      "additionalProperties": False
    }
  }
}



class ExtractSmiles():
    def __init__(self):
        self.storage = FileStorage(
            first_entry_file_name="../example_data/chemistry/matched_sample_10.json",
            #first_entry_file_name="/Users/lianghao/Desktop/dataflow_code/test_0901/example_data/chemistry/matched_sample_10.json",
            cache_path="./cache_all_17_24_gpt_5",
            file_name_prefix="math_QA",
            cache_type="json",
        )
        self.model_cache_dir = './dataflow_cache'
        self.llm_serving = APILLMServing_request(
                api_url="https://api.openai.com/v1/chat/completions",
                model_name="gemini-2.5-flash",
                max_workers=200,
        )
        self.prompt_smile_extractor = ExtractSmilesFromText(
            llm_serving = self.llm_serving, 
            prompt_template=ExtractSmilesFromTextPrompt(smiles_prompt),
        )
        self.smile_eval = EvaluateSmilesEquivalence()

    def forward(self):
        # Initial filters
        self.prompt_smile_extractor.run(
            storage = self.storage.step(),
            content_key = "text",
            abbreviation_key = "abbreviations",
            output_key = "synth_smiles"
        )
        self.smile_eval.run(
            storage = self.storage.step(),
        )


if __name__ == "__main__":
    # This is the entry point for the pipeline

    model = ExtractSmiles()
    model.forward()
