import torch
from tqdm import tqdm
from hashlib import md5, sha256
from xxhash import xxh3_128
from transformers import BertModel, BertTokenizer
from torch.nn.functional import normalize
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

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


@OPERATOR_REGISTRY.register()
class SemDeduplicator(OperatorABC):
    def __init__(self, eps: float = 0.05, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', model_cache_dir: str = './dataflow_cache', device: str = 'cuda'):
        self.logger = get_logger()
        self.eps = eps
        self.device = device
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=model_cache_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with eps = {self.eps}, model_name = {self.model_name}, model_cache_dir = {self.model_cache_dir}, device = {self.device}")
        
    def _compute_hash(self, text: str) -> str:
        return self.hash_func_dict[self.hash_func](text.encode('utf-8')).hexdigest()

    def run(self, storage: DataFlowStorage, input_keys: list = None, input_key: str = None, output_key: str = 'minhash_deduplicated_label'):
        if input_keys is None and input_key is None:
            self.logger.error(f"Need to specify either input_keys or input_key!")
            raise ValueError(f"Need to specify either input_keys or input_key!")
        if input_keys is not None and input_key is not None:
            self.logger.error(f"{self.__class__.__name__} only need one input args!")
            raise ValueError(f"{self.__class__.__name__} only need one input args!")
        if input_keys is not None:
            self.logger.info(f"Running {self.__class__.__name__} with input_keys = {input_keys} and output_key = {output_key}")
        else:
            self.logger.info(f"Running {self.__class__.__name__} with input_key = {input_key} and output_key = {output_key}")
        self.input_key = input_key
        self.input_keys = input_keys
        self.output_key = output_key
        seen_hashes = set()
        dataframe = storage.read("dataframe")
        texts = []
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            texts.append(text) 
        embeddings = get_text_embedding(texts, self.tokenizer, self.model, self.device)
        embeddings = normalize(torch.tensor(embeddings), dim=1)

        # Compute cosine similarity matrix
        cos_sim_matrix = compute_cos_sim_matrix(embeddings)
        cos_sim_matrix.fill_diagonal_(0)  # Set diagonal to 0 to avoid self-comparison
        cos_sim_matrix = torch.triu(cos_sim_matrix, diagonal=1)

        # Find pairs with similarity greater than or equal to the threshold
        similar_pairs = torch.where(cos_sim_matrix >= (1 - self.eps))

        labels = [1] * len(dataframe) 
        for idx in similar_pairs[1].tolist():
            labels[idx] = 0
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

