from dataflow.core import TextDeduplicator
from dataflow.utils.registry import PROCESSOR_REGISTRY
import torch
from transformers import BertModel, BertTokenizer
from torch.nn.functional import normalize
import json

def load_model(device, model_path):
    """
    Load the pretrained BERT model and tokenizer.

    Args:
        model_path (str): Path to the pretrained model.

    Returns:
        model, tokenizer: The loaded BERT model and tokenizer.
    """
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = model.to(device)
    model = model.eval()
    return model, tokenizer


def get_text_embedding(texts, tokenizer, model, device):
    """
    Compute text embeddings using the provided BERT model.

    Args:
        texts (list): List of texts to be embedded.
        tokenizer: Tokenizer for the model.
        model: The BERT model.

    Returns:
        np.ndarray: Embeddings for the input texts.
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Use mean pooling for sentence embeddings


def compute_cos_sim_matrix(embeddings):
    """
    Compute the cosine similarity matrix for the given embeddings.

    Args:
        embeddings (np.ndarray): Text embeddings.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    embeddings = torch.tensor(embeddings)
    embeddings = normalize(embeddings, dim=1)
    return embeddings @ embeddings.T


@PROCESSOR_REGISTRY.register()
class SemDeduplicator(TextDeduplicator):
    def __init__(self, args_dict: dict):
        super().__init__(args_dict)
        self.deduplicator_name = 'SemDeduplicator'
        self.eps = args_dict.get('eps', 0.05)
        self.model_path = 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = args_dict.get('device')
        self.model = BertModel.from_pretrained(self.model_path, cache_dir=args_dict.get('model_cache_dir')).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path, cache_dir=args_dict.get('model_cache_dir'))

    def dedup_func(self, dataset):
        """
        Deduplicate the dataset based on semantic similarity.

        Args:
            dataset: The input dataset to be deduplicated.

        Returns:
            TextSubset: A subset of the original dataset containing only unique samples.
        """
        # get texts
        texts = []
        for idx, sample in enumerate(dataset):
            if isinstance(dataset.keys, list):
                text = " ".join([str(sample[key]) for key in dataset.keys])
            else:
                text = str(sample[dataset.keys])
            texts.append(text)
        # Compute embeddings for the dataset texts
        embeddings = get_text_embedding(texts, self.tokenizer, self.model, self.device)
        embeddings = normalize(torch.tensor(embeddings), dim=1)
        print(json.dumps({"embeddings": embeddings.tolist()}))
        return json.dumps({"embeddings": embeddings.tolist()})
