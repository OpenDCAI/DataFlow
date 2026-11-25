"""Optional local CLIP serving resource for near-duplicate detection."""
from pathlib import Path
from PIL import Image, ImageOps
from .preprocessing import projected_clip_output


class LocalCLIPEncoder:
    """Optional real CLIP adapter. Loads only explicitly supplied local weights."""
    def __init__(self, model_path, device='cpu'):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch, self.device = torch, device
        self.model_id = str(Path(model_path).resolve())
        self.processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        self.mocked = False

    def encode(self, path):
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
            values = self.processor(images=image, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            output = self.model.get_image_features(**values)
            features = projected_clip_output(output)
        return features[0].float().cpu().numpy()


    def cleanup(self):
        self.model.to('cpu')
