from dataflow.core import TextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.Eval.Text.models.Superfiltering.data_analysis import get_perplexity_and_embedding_whole_text, get_perplexity_and_embedding_part_text
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

# Superfiltering instruction quality (ifd) evaluation
# cited from: Superfiltering: Weak-to-Strong Data Filtering for Fast Instruction-Tuning
@MODEL_REGISTRY.register()
class SuperfilteringScorer(TextScorer):
    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.device = args_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = args_dict.get('model_name')
        self.model_cache_dir = args_dict.get('model_cache_dir')
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device, cache_dir=self.model_cache_dir, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.model_cache_dir)
        self.prompt = args_dict.get('prompt', 'none')
        self.max_length = args_dict.get('max_length', 512) 
        self.batch_size = 1
        self.score_type = float 
        self.data_type = 'text' 
        self.score_name = 'SuperfilteringScore' 

    def evaluate_batch(self, batch):
        PROMPT_DICT_NONE = {
            "prompt_input": (
                "{instruction}\n{input}\n"
            ),
            "prompt_no_input": (
                "{instruction}\n"
            ),
        }

        if self.prompt == 'none':
            prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
            prompt_input = PROMPT_DICT_NONE["prompt_input"]

        scores = []

        instruction = batch.get('instruction', [''])[0]
        output = batch.get('output', [''])[0]
        input_text = batch.get('input', [''])[0] if 'input' in batch else ''
        if input_text == '':
            temp_dict = {'instruction': instruction}
            prompt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use
        else:
            temp_dict = {'instruction': instruction, 'input': input_text}
            prompt_to_use = prompt_input.format_map(temp_dict)
            whole_text = prompt_to_use + output
            instruction = prompt_to_use

        instruction_input_ids = self.tokenizer.encode(instruction, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        instruction_len = instruction_input_ids.shape[1]

        if output == '':
            score = None
        else:
            ppl_out_alone, _ = get_perplexity_and_embedding_whole_text(self.tokenizer, self.model, output, self.max_length - instruction_len + 1, self.device)
            ppl_out_condition, _ = get_perplexity_and_embedding_part_text(self.tokenizer, self.model, whole_text, output, self.max_length, self.device)

            if ppl_out_alone != 0:
                score = ppl_out_condition / ppl_out_alone
            else:
                score = 0

        if score != score:  
            score = None

        scores.append(score)

        return scores