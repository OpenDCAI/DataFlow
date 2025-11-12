from dataflow.core.prompt import PromptABC
from dataflow.utils.registry import PROMPT_REGISTRY
@PROMPT_REGISTRY.register()
class ExtractSmilesFromTextPrompt(PromptABC):
    def __init__(self, prompt_template = None):
        if prompt_template is None:
            self.prompt_template = """Extract the monomer/small molecule information from the text and format it as a structured JSON object.
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
        else:
            self.prompt_template = prompt_template
    def build_prompt(self, target_monomers: str) -> str:
        target_prompt = "\nHere give you some monomers' abbreviation or full name, please only extract the information of these monomers. This rule have priority over the other rules. Here are the specific monomers: " + str(target_monomers)
        return self.prompt_template + target_prompt