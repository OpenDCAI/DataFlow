# Modified according to https://github.com/Yebin46/FLEUR/blob/master/fleur.py and https://docs.vllm.ai/en/stable/getting_started/examples/offline_inference_vision_language.html

import numpy as np
import torch
import math
from vllm import LLM, SamplingParams
import re
import os

from dataflow.core.scorer import ImageTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from ...utils.image_utils import fleur_collate 


@MODEL_REGISTRY.register()
class FleurScorer(ImageTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        model_name = args_dict["hf_model_path"].split("/")[-1]
        model_cache_path=os.path.join(args_dict["model_cache_dir"], model_name)
        try:
            self.model = LLM(model=model_cache_path)
        except:
            download_hf_model(model_cache_path, args_dict["hf_model_path"])
            self.model = LLM(model=model_cache_path)

        tokenizer = self.model.get_tokenizer()
        self.rate2token = {s : tokenizer.encode(str(s))[-1] for s in range(10)}
        self.sampling_params = SamplingParams(temperature=0.2,
                                            max_tokens=15,
                                            stop_token_ids=None,
                                            logprobs=10
                                            )

        self.prompt_template = 'USER: <image>\nYour task is to evaluate and rate the caption on a scale of 0.0 to 1.0 based on the given Grading Criteria. (Print Real Number Score ONLY)\n\nGrading Criteria:\n\n0.0: The caption does not describe the image at all.\n1.0: The caption accurately and clearly describes the image.\n\nCaption: {}\n\nScore(Choose a rating from 0.0 to 1.0):\nASSISTANT:'
        self.collate_fn = fleur_collate
        self.data_type = "image_caption"
        self.scorer_name = "FleurScorer"
        

    def evaluate_batch(self, sample):
        inputs = [
            {
            "prompt": self.prompt_template.format(text),
            "multi_modal_data": {
                "image": image
            }, 
            } for image, text in zip(sample[0], sample[1])
        ]

        scores = np.zeros(len(inputs))
        generate_outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        for idx, o in enumerate(generate_outputs):
            generated_text = o.outputs[0].text.strip()
            output_ids = torch.tensor(o.outputs[0].token_ids)
            output_logprobs = o.outputs[0].logprobs

            try:
                dotsnumbersdots = re.sub(f'[^\d\.]', '', generated_text)
                numbersdots = re.sub(f'^\.+', '', dotsnumbersdots)
                numbers = re.sub(r'\.+$', '', numbersdots)
                score_check = float(numbers)

                if 0 > score_check or 1 < score_check:
                    continue
                
                if score_check < 1.0:
                    num_index_in_score = str(score_check).index('.') + 1
                    find_num = int(str(score_check)[num_index_in_score])
                    num_index_in_token = (output_ids == self.rate2token[find_num]).nonzero().squeeze()
                    if len(num_index_in_token.shape) > 0: # if there is a duplication, choose one: e.g., 0.0 -> select the second 0 (after "."), 0.66 -> select the first 6
                        if find_num == 0:
                            num_index_in_token = num_index_in_token[1]
                        else:
                            num_index_in_token = num_index_in_token[0]

                    probs = output_logprobs[num_index_in_token]
                    
                    score = 0.
                    for rate, token in self.rate2token.items(): # score smoothing
                        score += math.exp(probs[token].logprob) * rate * 0.1
                        
                    if len(str(score_check)) > 3: # second decimal place case, 0 < score_check < 1.0
                        num2_index_in_score = str(score_check).index('.') + 2
                        find_num2 = int(str(score_check)[num2_index_in_score])
                        num2_index_in_token = (output_ids == self.rate2token[find_num2]).nonzero().squeeze()
                        if len(num2_index_in_token.shape) > 0: # if there is a duplication, choose the second one.
                            num2_index_in_token = num2_index_in_token[1]
                        probs2 = output_logprobs[num2_index_in_token]
                    
                        for rate, token in self.rate2token.items():
                            score += math.exp(probs2[token].logprob) * rate * 0.01
                else: # only 1.0 case
                    num_index_in_token = (output_ids == self.rate2token[1]).nonzero().squeeze()
                    probs = output_logprobs[num_index_in_token]
                    score = 0.9 * math.exp(probs[self.rate2token[0]].logprob) + math.exp(probs[self.rate2token[1]].logprob)
                scores[idx] = score
            except:
                print("Error!")
                scores[idx] = -1

        return scores