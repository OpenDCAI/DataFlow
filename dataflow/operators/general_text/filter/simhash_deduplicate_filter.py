from tqdm import tqdm
from simhash import Simhash
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage
from dataflow.utils.registry import OPERATOR_REGISTRY

def get_similarity(simhash, another_simhash):
    max_hashbit = max(len(bin(simhash.value)), len(bin(another_simhash.value)))
    distince = simhash.distance(another_simhash)
    similar = 1 - distince / max_hashbit
    return similar

@OPERATOR_REGISTRY.register()
class SimHashDeduplicateFilter(OperatorABC):
    def __init__(self, fingerprint_size: int = 64, bound: float = 0.1):
        self.logger = get_logger()
        self.fingerprint_size = fingerprint_size
        self.bound = bound
        self.logger.info(f"Initializing {self.__class__.__name__} with fingerprint_size = {self.fingerprint_size}, bound = {self.bound}...")
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用SimHash算法通过汉明距离识别相似文本，执行近似去重操作。将文本转换为固定长度的指纹，通过计算指纹间的汉明距离判断文本相似度。\n"
                "相比语义去重速度更快，适合大规模数据集的快速去重预处理，尤其适用于检测字符层面相似的文本。\n"
                "输入参数：\n"
                "- fingerprint_size：指纹长度，默认为64位\n"
                "- bound：相似度阈值，值越小表示允许的相似度越低，默认为0.1（即相似度大于0.9视为重复）\n"
                "- input_keys：多个输入字段名列表，与input_key二选一\n"
                "- input_key：单个输入字段名，与input_keys二选一\n"
                "- output_key：去重结果字段名，默认为'minhash_deduplicated_label'\n"
                "输出参数：\n"
                "- 过滤后的DataFrame，仅保留相似性低于阈值的唯一样本（标记为1的样本）\n"
                "- 返回包含去重结果字段名的列表，用于后续算子引用"
            )
        elif lang == "en":
            return (
                "Detect similar text via SimHash algorithm and Hamming distance for near deduplication. Convert text to fixed-length fingerprints and determine text similarity by calculating Hamming distance between fingerprints.\n"
                "Faster than semantic deduplication, suitable for fast deduplication preprocessing of large-scale datasets, especially for detecting character-level similar texts.\n"
                "Input Parameters:\n"
                "- fingerprint_size: Fingerprint length, default is 64 bits\n"
                "- bound: Similarity threshold, smaller values allow lower similarity, default is 0.1 (similarity > 0.9 is considered duplicate)\n"
                "- input_keys: List of multiple input field names, alternative to input_key\n"
                "- input_key: Single input field name, alternative to input_keys\n"
                "- output_key: Deduplication result field name, default is 'minhash_deduplicated_label'\n"
                "Output Parameters:\n"
                "- Filtered DataFrame containing only unique samples with similarity below threshold (samples marked as 1)\n"
                "- List containing deduplication result field name for subsequent operator reference"
            )
        else:
            return "Near deduplication by detecting text similarity using SimHash algorithm and Hamming distance."

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
        simhashes = []
        labels = [0] * len(dataframe)
        for idx, sample in tqdm(enumerate(dataframe.to_dict(orient='records')), desc=f"Implementing {self.__class__.__name__}", total=len(dataframe)):
            if input_keys is not None and len(input_keys) > 1:
                text = '\n'.join([f"{k}:\n{sample[k]}" for k in input_keys])
            else:
                text = sample[self.input_key]
            simhash = Simhash(text, f=self.fingerprint_size)
            if all(get_similarity(simhash, another_simhash) < 1 - self.bound for another_simhash in simhashes):
                labels[idx] = 1
                simhashes.append(simhash)
        dataframe[self.output_key] = labels
        filtered_dataframe = dataframe[(dataframe[self.output_key] > 0)]
        output_file = storage.write(filtered_dataframe)
        self.logger.info(f"Deduplication completed. Total unique items: {sum(labels)}")
        return [self.output_key,]
        
        

        
        

