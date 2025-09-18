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
class SemDeduplicateFilter(OperatorABC):
    def __init__(self, eps: float = 0.05, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', model_cache_dir: str = './dataflow_cache', device: str = 'cuda'):
        self.logger = get_logger()
        self.eps = eps
        self.device = device
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=model_cache_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with eps = {self.eps}, model_name = {self.model_name}, model_cache_dir = {self.model_cache_dir}, device = {self.device}")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "基于BERT语义相似度识别语义重复文本，执行近似去重操作。通过计算文本嵌入向量间的余弦相似度，识别语义相似的文本并保留唯一样本。\n"
                "支持多字段组合作为去重依据，可有效去除内容相似但表述不同的重复数据，提高数据集多样性。\n"
                "输入参数：\n"
                "- eps：相似度阈值，值越小表示允许的相似度越低，默认为0.05（即余弦相似度大于0.95视为重复）\n"
                "- model_name：预训练模型名称，默认为'sentence-transformers/all-MiniLM-L6-v2'\n"
                "- model_cache_dir：模型缓存目录，默认为'./dataflow_cache'\n"
                "- device：模型运行设备，默认为'cuda'\n"
                "- input_keys：多个输入字段名列表，与input_key二选一\n"
                "- input_key：单个输入字段名，与input_keys二选一\n"
                "- output_key：去重结果字段名，默认为'minhash_deduplicated_label'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留语义不重复的样本（标记为1的样本）\n"
                "- 返回包含去重结果字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Identify semantically duplicate text using BERT embeddings for near deduplication. Calculate cosine similarity between text embedding vectors to detect semantically similar texts and retain unique samples.\n"
                "Supports multiple field combinations as deduplication criteria, effectively removing duplicate data with similar content but different expressions to improve dataset diversity.\n"
                "Input Parameters:\n"
                "- eps: Similarity threshold, smaller values allow lower similarity, default is 0.05 (cosine similarity > 0.95 is considered duplicate)\n"
                "- model_name: Pretrained model name, default is 'sentence-transformers/all-MiniLM-L6-v2'\n"
                "- model_cache_dir: Model cache directory, default is './dataflow_cache'\n"
                "- device: Model running device, default is 'cuda'\n"
                "- input_keys: List of multiple input field names, alternative to input_key\n"
                "- input_key: Single input field name, alternative to input_keys\n"
                "- output_key: Deduplication result field name, default is 'minhash_deduplicated_label'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only semantically unique samples (samples marked as 1)\n"
                "- List containing deduplication result field name for subsequent operator reference"
            )
        else:
            return "Near deduplication by identifying semantically similar content using BERT embeddings."

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
        
        

        
        

