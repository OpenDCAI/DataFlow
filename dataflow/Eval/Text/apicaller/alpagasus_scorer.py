from openai import OpenAI
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# Alpagasus instruction quality evaluation
# cited from: AlpaGasus: Training A Better Alpaca with Fewer Data
@MODEL_REGISTRY.register()
class AlpagasusScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.api_key = args_dict.get('api_key')
        self.model = args_dict.get('model')
        self.dimension = args_dict.get('dimension')
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'AlpagasusScore' 
        self.client = OpenAI(api_key=self.api_key)

        self.system_prompt_template = """
        We would like to request your feedback on the performance of AI assistant in response to the instruction and the given input displayed following.
        Instruction: {instruction}
        Input: {input}
        Response: {response}
        """
        self.user_prompt_template = """
        Please rate according to the {dimension} of the response to the instruction and the input. Each assistant
        receives a score on a scale of 0 to 5, where a higher score indicates a higher level of the {dimension}. Please
        first output a single line containing the value indicating the scores. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.
        """

    def get_score(self, instruction, input_text, response):
        system_prompt = self.system_prompt_template.format(instruction=instruction, input=input_text, response=response)
        user_prompt = self.user_prompt_template.format(dimension=self.dimension)

        api_response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        score_line = api_response.choices[0].message.content.strip().split("\n")[0]
        
        score = float(score_line.split()[0])

        return score


    def evaluate_batch(self, batch):
        results = [] 
        instruction = batch.get('instruction')
        input_text = batch.get('input')
        response = batch.get('output')
        for i in range(self.batch_size): 
            score = self.get_score(instruction[i], input_text[i], response[i])
            results.append(score)

        return results 
