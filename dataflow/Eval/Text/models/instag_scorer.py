import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY

# Instag instruction complexity evaluation
# cited from: #INSTAG: INSTRUCTION TAGGING FOR ANALYZING SUPERVISED FINE-TUNING OF LARGE LANGUAGE MODELS
@MODEL_REGISTRY.register()
class InstagScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.model_path = args_dict.get('model_path')
        self.max_new_tokens = args_dict.get('max_new_tokens', 50)
        self.model_cache_dir = args_dict.get('model_cache_dir')  
        self.temperature = args_dict.get('temperature', 1.0)
        self.do_sample = args_dict.get('do_sample', False)
        self.num_return_sequences = args_dict.get('num_return_sequences', 1)
        self.return_dict_in_generate = args_dict.get('return_dict_in_generate', True)
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=self.model_cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, cache_dir=self.model_cache_dir).to(self.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.batch_size = 1
        self.score_type = float  
        self.data_type = 'text' 
        self.score_name = 'InstagScore'  

    def make_prompt(self, query):
        prompt = f"Please identify tags of user intentions in the following user query and provide an explanation for each tag. Please respond in the JSON format {{\"tag\": str, \"explanation\": str}}.\nUser query: {query}"
        messages = [("USER", prompt), ("ASSISTANT", None)]
        seps = [" ", "</s>"]
        ret = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions." + seps[0]
        for i, (role, message) in enumerate(messages):
            if message:
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ":"
        return ret

    def inference_batch(self, queries):
        input_strs = [self.make_prompt(query) for query in queries]
        input_tokens = self.tokenizer(input_strs, return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            input_tokens = {key: value.to(self.device) for key, value in input_tokens.items()}

        output = self.model.generate(
            input_tokens['input_ids'],
            temperature=self.temperature,
            do_sample=self.do_sample,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=self.num_return_sequences,
            return_dict_in_generate=self.return_dict_in_generate,
        )
        
        num_input_tokens = input_tokens["input_ids"].shape[1]
        output_tokens = output.sequences
        generated_tokens = output_tokens[:, num_input_tokens:]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        json_outputs = []
        for generated_text in generated_texts:
            string_output = generated_text.strip()
            try:
                json_output = json.loads(string_output)
            except json.JSONDecodeError:
                json_output = {"error": "JSON decode error."}
            json_outputs.append(json_output)
        
        return json_outputs

    def evaluate_batch(self, batch):
        queries = batch.get('instruction', ['']) 
        json_outputs = self.inference_batch(queries)
        
        scores = []
        for json_output in json_outputs:
            if isinstance(json_output, list):
                complexity_score = len(json_output)
            else:
                complexity_score = 0
            scores.append(int(complexity_score))
        
        return scores
