import numpy as np
import torch
from vllm import LLM, SamplingParams
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dataflow.core.scorer import ImageTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from dataflow.utils.image_utils import download_hf_model
from dataflow.utils.image_utils import fleur_collate_fn 

@MODEL_REGISTRY.register()
class VisualDialogScorer(ImageTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        model_name = args_dict["hf_model_path"].split("/")[-1]
        model_cache_path=os.path.join(args_dict["model_cache_dir"], model_name)
        try:
            self.model = LLM(model=model_cache_path, trust_remote_code=True, tensor_parallel_size=args_dict["tensor_parallel_size"], gpu_memory_utilization=args_dict["gpu_memory_utilization"])
        except:
            download_hf_model(model_cache_path, args_dict["hf_model_path"])
            self.model = LLM(model=model_cache_path, trust_remote_code=True, tensor_parallel_size=args_dict["tensor_parallel_size"], gpu_memory_utilization=args_dict["gpu_memory_utilization"])

        tokenizer = self.model.get_tokenizer()
        self.sampling_params = SamplingParams(temperature=0.2,
                                            max_tokens=64,
                                            stop_token_ids=None,
                                            logprobs=10
                                            )
        self.correct_id = tokenizer.encode("correct")[-1]
        self.incorrect_id = tokenizer.encode("incorrect")[-1]
        self.prompt_template = args_dict["prompt_template"]
        self.collate_fn = fleur_collate_fn
        self.data_type = "image_caption"
        self.scorer_name = "VisualDialogScorer"

    def evaluate_batch(self, batch):
        inputs = [
            {
            "prompt": self.prompt_template.format(text),
            "multi_modal_data": {
                "image": image
            }, 
            } for image, text in zip(batch[0], batch[1])
        ]

        scores = np.zeros(len(inputs))
        generate_outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
        scores = np.zeros(len(generate_outputs))
        for idx, o in enumerate(generate_outputs):
            try:
                # generated_text = o.outputs[0].text.strip()
                # print(generated_text)
                output_ids = torch.tensor(o.outputs[0].token_ids)
                output_logprobs = o.outputs[0].logprobs

                correct_index = (output_ids == self.correct_id).nonzero()
                correct_index = correct_index[0] if len(correct_index) > 0 else None
                incorrect_index = (output_ids == self.incorrect_id).nonzero()
                incorrect_index = incorrect_index[0] if len(incorrect_index) > 0 else None
                if correct_index is None and incorrect_index is None:
                    score = -1
                else:
                    final_index = correct_index if correct_index is not None else incorrect_index
                    
                    probs = output_logprobs[final_index]
                    logprob_correct = probs[self.correct_id].logprob if correct_index is not None else -1000
                    score = logprob_correct
                scores[idx] = score
            except:
                scores[idx] = -2

        return scores
