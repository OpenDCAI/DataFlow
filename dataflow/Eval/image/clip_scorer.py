import torch
import clip
import os

from dataflow.core.scorer import ImageTextScorer
from dataflow.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ClipScorer(ImageTextScorer):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        model, preprocess = clip.load(name="ViT-B/32", device=args_dict["device"], download_root=os.path.join(args_dict["model_cache_dir"], "clip"))
        self.model = model.eval()
        self.image_preprocessor = preprocess
        self.text_preprocessor = clip.tokenize
        self.device = args_dict["device"]
        self.data_type = "image_caption"
        self.scorer_name = "ClipScorer"
    
    def evaluate_batch(self, sample):
        image_features = self.model.encode_image(sample[0].to(self.device)) # [batch_size, dim]
        text_features = self.model.encode_text(sample[1].squeeze(1).to(self.device))
        
        # normalize the features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        scores = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(2)).squeeze(1).squeeze(1)*100 # [batch_size, 1, 1]->[batch_size]
        return scores
    