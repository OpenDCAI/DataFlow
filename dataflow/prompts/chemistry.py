
class ExtractSmilesFromTextPrompt:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template
    
    def build_prompt(self, target_monomers: str) -> str:
        target_prompt = "\nHere give you some monomers' abbreviation or full name, please only extract the information of these monomers. This rule have priority over the other rules. Here are the specific monomers: " + str(target_monomers)
        return self.prompt_template + target_prompt