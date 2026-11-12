import numpy as np
import torch
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from torch.nn.functional import normalize
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

DEFAULT_BATCH_SIZE = 256
DEFAULT_BLOCK_SIZE = 1024

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


def get_text_embedding(texts, tokenizer, model, device, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Compute text embeddings using the provided BERT model.

    Args:
        texts (list): List of texts to be embedded.
        tokenizer: Tokenizer for the model.
        model: The BERT model.
        device: Device the model lives on.
        batch_size (int): Number of texts per forward pass.

    Returns:
        np.ndarray: Embeddings for the input texts.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling for sentence embeddings
        embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())
    if not embeddings:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)


def compute_cos_sim_matrix(embeddings):
    """
    Compute the cosine similarity matrix for the given embeddings.

    Kept for callers that already hold a small number of embeddings; the
    operator itself scores in blocks instead, see `find_duplicate_labels`.

    Args:
        embeddings (np.ndarray): Text embeddings.

    Returns:
        np.ndarray: Cosine similarity matrix.
    """
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
    embeddings = normalize(embeddings, dim=1)
    return embeddings @ embeddings.T


def _unit_rows(embeddings):
    """L2-normalise every row so that a dot product is a cosine similarity."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}.")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # an all-zero row is similar to nothing
    return embeddings / norms


def find_duplicate_labels(embeddings, threshold: float, block_size: int = DEFAULT_BLOCK_SIZE):
    """
    Label near-duplicate rows without materialising the full similarity matrix.

    A row is labelled ``0`` when it is at least ``threshold`` similar to an
    earlier row that is itself kept, and ``1`` otherwise. Comparing against kept
    rows only keeps a chain of near-duplicates from wiping out its own tail: for
    A ~ B ~ C where A is not similar to C, B is dropped while A and C both
    survive. The previous implementation dropped the second element of every
    similar pair, so that same chain lost C as well.

    Rows are scored `block_size` at a time, so peak memory is O(block_size ** 2)
    instead of the O(N ** 2) of a full cosine matrix (100k rows would need 38 GB).

    Args:
        embeddings (np.ndarray): Text embeddings, one row per record.
        threshold (float): Cosine similarity at or above which two rows are duplicates.
        block_size (int): Number of rows scored per block.

    Returns:
        np.ndarray: Integer labels of length N, ``1`` for kept rows.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be a positive integer, got {block_size}.")
    unit = _unit_rows(embeddings)
    num_rows = unit.shape[0]
    labels = np.ones(num_rows, dtype=int)
    if num_rows == 0:
        return labels

    for start in range(0, num_rows, block_size):
        stop = min(start + block_size, num_rows)
        rows = unit[start:stop]

        # Rows of this block that are already covered by a kept row from an earlier block.
        covered = np.zeros(stop - start, dtype=bool)
        for col_start in range(0, start, block_size):
            col_stop = min(col_start + block_size, start)
            sims = rows @ unit[col_start:col_stop].T
            sims[:, labels[col_start:col_stop] == 0] = -np.inf
            covered |= sims.max(axis=1) >= threshold

        # Resolve the block against itself, in index order, so that a row dropped
        # here cannot be the reason for dropping one of its own successors.
        intra_block_sims = rows @ rows.T
        for offset in range(stop - start):
            if covered[offset]:
                labels[start + offset] = 0
                continue
            if offset:
                kept_earlier = labels[start:start + offset] == 1
                if np.any(intra_block_sims[offset, :offset][kept_earlier] >= threshold):
                    labels[start + offset] = 0

    return labels


@OPERATOR_REGISTRY.register()
class SemDeduplicateFilter(OperatorABC):
    def __init__(self, eps: float = 0.05, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', model_cache_dir: str = './dataflow_cache', device: str = 'cuda', batch_size: int = DEFAULT_BATCH_SIZE, block_size: int = DEFAULT_BLOCK_SIZE):
        self.logger = get_logger()
        self.eps = eps
        self.device = device
        self.model_name = model_name
        self.model_cache_dir = model_cache_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.model = BertModel.from_pretrained(self.model_name, cache_dir=model_cache_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=model_cache_dir)
        self.logger.info(f"Initializing {self.__class__.__name__} with eps = {self.eps}, model_name = {self.model_name}, model_cache_dir = {self.model_cache_dir}, device = {self.device}, batch_size = {self.batch_size}, block_size = {self.block_size}")
    
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
                "- batch_size：每次前向计算的文本条数，默认为256\n"
                "- block_size：每次计算相似度的行数，默认为1024；越大越快、占用内存越多\n"
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
                "- batch_size: Number of texts per forward pass, default is 256\n"
                "- block_size: Number of rows scored per similarity block, default is 1024 (larger is faster and uses more memory)\n"
                "- input_keys: List of multiple input field names, alternative to input_key\n"
                "- input_key: Single input field name, alternative to input_keys\n"
                "- output_key: Deduplication result field name, default is 'minhash_deduplicated_label'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only semantically unique samples (samples marked as 1)\n"
                "- List containing deduplication result field name for subsequent operator reference"
            )
        else:
            return "Near deduplication by identifying semantically similar content using BERT embeddings."

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
        dataframe = storage.read("dataframe")
        texts = []
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key if input_keys is None else input_keys[0]]
            texts.append(text) 

        if len(texts) == 0:
            dataframe[self.output_key] = []
            storage.write(dataframe)
            self.logger.info("Deduplication skipped: the input dataframe is empty.")
            return [self.output_key,]

        embeddings = get_text_embedding(texts, self.tokenizer, self.model, self.device, batch_size=self.batch_size)
        labels = find_duplicate_labels(embeddings, 1 - self.eps, block_size=self.block_size)

        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
