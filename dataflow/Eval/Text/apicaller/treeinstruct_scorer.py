from openai import OpenAI
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# Tree-Instruct instruction complexity evaluation
# cited from: Tree-Instruct: A Preliminary Study of the Intrinsic Relationship between Complexity and Alignment
@MODEL_REGISTRY.register()
class TreeinstructScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.api_key = args_dict.get('api_key')
        self.model = args_dict.get('model')
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'TreeinstructScore' 
        self.client = OpenAI(api_key=self.api_key)

        self.system_prompt_template = """
        You are an instruction rewriter. You need to parse a given user instruction into a TREE structure following Semantic Parsing in the natural language processing field.
        Procedure:
        step-1: Parse the old “instruction” to a TREE-1 through Semantic Parsing in the natural language processing field. 
        Count and return the number of nodes in TREE-1.
        Old instruction: “{instruction}”
        """
        self.user_prompt_template = """
        Please count and return ONLY the number of nodes in TREE-1. This number represents the complexity of the original instruction. 
        For example: 4
        """

    def get_complexity_score(self, instruction):
        system_prompt = self.system_prompt_template.format(instruction=instruction)
        user_prompt = self.user_prompt_template

        api_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        score_line = api_response.choices[0].message.content.strip()
        try:
            score = int(score_line)
        except ValueError:
            score = 0  

        return score
    
    def evaluate_batch(self, batch):
        results = []
        instruction = batch.get('instruction')
        for i in range(self.batch_size): 
            score = self.get_complexity_score(instruction[i])
            results.append(score)

        return results
