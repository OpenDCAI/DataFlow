import numpy as np
import os
import torch
import subprocess
from dataflow.Eval.image.longclip.model import longclip
from dataflow.core.scorer import ImageTextScorer
from dataflow.utils.registry import MODEL_REGISTRY
from ...utils.utils import download_model_from_hf


@MODEL_REGISTRY.register()
class LongClipScorer(ImageTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        model_cache_dir = os.path.join(args_dict["model_cache_dir"], "longclip")
        model_cache_path=os.path.join(model_cache_dir, f"longclip-{args_dict['model_size']}.pt")
        try:
            model, preprocess = longclip.load(name=model_cache_path, device=args_dict["device"])
        except:
            download_model_from_hf("BeichenZhang/LongCLIP-" + args_dict["model_size"], model_cache_dir)
            model, preprocess = longclip.load(name=model_cache_path, device=args_dict["device"])

        self.model = model.eval()
        self.image_preprocessor = preprocess
        self.text_preprocessor = longclip.tokenize
        self.device = args_dict["device"]
        self.data_type = "image_caption"
        self.scorer_name = "LongClipScorer"


    def evaluate_batch(self, sample):
        image_features = self.model.encode_image(sample[0].to(self.device)) # [batch_size, dim]
        text_features = self.model.encode_text(sample[1].squeeze(1).to(self.device))

        scores = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(2)).squeeze(1).squeeze(1) # [batch_size, 1, 1]->[batch_size]
        return scores
    