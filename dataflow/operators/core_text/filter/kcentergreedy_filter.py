import numpy as np
import pandas as pd
import torch
from torch import Tensor
from typing import List, Optional
import torch.nn.functional as F
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC, LLMServingABC

class KCenterGreedy:
    """Implements k-center-greedy method.

    Args:
        embedding (Tensor): Embedding vector extracted from a LLM
        sampling_ratio (float): Ratio to choose coreset size from the embedding size.

    Example:
        >>> embedding.shape
        torch.Size([219520, 1536])
        >>> sampler = KCenterGreedy(embedding=embedding)
        >>> sampled_idxs = sampler.select_coreset_idxs()
        >>> coreset = embedding[sampled_idxs]
        >>> coreset.shape
        torch.Size([219, 1536])
    """

    def __init__(self, embedding: Tensor, sampling_ratio: float) -> None:
        self.embedding = embedding
        self.coreset_size = int(embedding.shape[0] * sampling_ratio)
        # self.model = SparseRandomProjection(eps=0.9)

        self.features: Tensor
        self.min_distances: Tensor = torch.tensor([])
        self.n_observations = self.embedding.shape[0]

    def reset_distances(self) -> None:
        """Reset minimum distances."""
        self.min_distances = torch.tensor([])

    def update_distances(self, cluster_centers: List[int]) -> None:
        """Update min distances given cluster centers.

        Args:
            cluster_centers (List[int]): indices of cluster centers
        """

        for center in cluster_centers:
            distance = F.pairwise_distance(
                self.features, self.features[center], p=2, eps=0
            ).reshape(-1, 1)

            if self.min_distances.shape[0] == 0:
                self.min_distances = distance
            else:
                self.min_distances = torch.minimum(self.min_distances, distance)

    def get_new_idx(self) -> int:
        """Get index value of a sample.

        Based on minimum distance of the cluster

        Returns:
            int: Sample index
        """

        if isinstance(self.min_distances, Tensor):
            idx = int(torch.argmax(self.min_distances).item())
        else:
            raise ValueError(f"self.min_distances must be of type Tensor. Got {type(self.min_distances)}")

        return idx

    def select_coreset_idxs(self, selected_idxs: Optional[List[int]] = None) -> List[int]:
        """Greedily form a coreset to minimize the maximum distance of a cluster.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
          Up to coreset_size distinct new indices, excluding selected_idxs.
        """

        selected_idxs = list(dict.fromkeys(selected_idxs or []))
        self.reset_distances()
        n_samples = min(self.coreset_size, self.n_observations - len(selected_idxs))
        if n_samples <= 0:
            return []

        self.features = self.embedding.reshape(self.n_observations, -1)
        if selected_idxs:
            self.update_distances(cluster_centers=selected_idxs)
            self.min_distances[selected_idxs] = -float("inf")
        else:
            idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
            self.update_distances(cluster_centers=[idx])

        selected_coreset_idxs: List[int] = []
        for step in range(n_samples):
            idx = self.get_new_idx()
            if step + 1 < n_samples:
                self.update_distances(cluster_centers=[idx])
            self.min_distances[idx] = -float("inf")
            selected_coreset_idxs.append(idx)

        return selected_coreset_idxs

    def sample_coreset(self, selected_idxs: Optional[List[int]] = None) -> Tensor:
        """Select coreset from the embedding.

        Args:
            selected_idxs: index of samples already selected. Defaults to an empty set.

        Returns:
            Tensor: Output coreset

        Example:
            >>> embedding.shape
            torch.Size([219520, 1536])
            >>> sampler = KCenterGreedy(...)
            >>> coreset = sampler.sample_coreset()
            >>> coreset.shape
            torch.Size([219, 1536])
        """

        idxs = self.select_coreset_idxs(selected_idxs)
        coreset = self.embedding[idxs]

        return coreset

@OPERATOR_REGISTRY.register()
class KCenterGreedyFilter(OperatorABC):
    def __init__(self, num_samples: int, embedding_serving : LLMServingABC = None):
        if (
            isinstance(num_samples, bool)
            or not isinstance(num_samples, int)
            or num_samples < 0
        ):
            raise ValueError("num_samples must be a non-negative integer")
        self.num_samples = num_samples
        self.embedding_serving = embedding_serving
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子利用 K-Center Greedy 算法从海量文档片段中挑选出具有代表性的样本（Coreset），常用于多样化种子 QA 对的生成。\n\n"
                "输入参数:\n"
                "- input_key: 包含文档内容的字段名（默认 'content'）。\n"
                "- embedding_serving: 嵌入模型服务实例，用于将文本转换为向量。\n"
                "- num_samples: 最终需要保留的文档片段数量（非负整数，最多为输入行数）。\n\n"
                "输出:\n"
                "- 过滤后的数据集，仅包含被选中的代表性文档片段。"
            )
        elif lang == "en":
            return (
                "This operator uses the K-Center Greedy algorithm to select representative samples (coreset) "
                "from a large pool of document fragments, typically used for generating diverse seed QA pairs.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the document content (default: 'content').\n"
                "- embedding_serving: Instance of embedding serving for text-to-vector conversion.\n"
                "- num_samples: Non-negative target count, capped at the input row count.\n\n"
                "Output:\n"
                "- A filtered dataset containing only the selected representative document fragments."
            )
        else:
            return "KCenterGreedyFilter selects representative document fragments using a greedy coreset strategy."
    
    def _validate_dataframe(self, dataframe: pd.DataFrame):
        required_keys = [self.input_key]
        forbidden_keys = []

        missing = [k for k in required_keys if k not in dataframe.columns]
        conflict = [k for k in forbidden_keys if k in dataframe.columns]

        if missing:
            self.logger.error(f"Missing required column(s): {missing}")
        if conflict:
            self.logger.error(f"The following column(s) already exist and would be overwritten: {conflict}")
        missing_keys = [key for key in required_keys if key not in dataframe.columns]

        if missing_keys:
            self.logger.error(f"The following required columns are missing from the dataframe: {missing_keys}")
    
    def run(
            self,
            storage:DataFlowStorage,
            input_key: str = "content",
            ) -> list:
        '''
        Execute the answer format filter process
        '''
        self.input_key = input_key
        dataframe = storage.read("dataframe")
        self._validate_dataframe(dataframe)

        texts = dataframe[self.input_key].tolist()
        n_samples = min(self.num_samples, len(texts))
        if n_samples == 0:
            output_file = storage.write(dataframe.iloc[:0])
            self.logger.info(f"Results saved to {output_file}")
            return [self.input_key]

        indexes =  np.zeros(len(dataframe)).astype(int)
        

        embeddings_list = self.embedding_serving.generate_embedding_from_input(texts)
        embeddings = torch.tensor(embeddings_list)

        sampler = KCenterGreedy(embedding=embeddings, sampling_ratio=n_samples / len(texts))
        # Preserve the integer budget instead of rounding through a ratio.
        sampler.coreset_size = n_samples
        chooss_indexes = sampler.select_coreset_idxs()


        for index in chooss_indexes:
            indexes[index] = 1

        dataframe = dataframe[np.array(indexes) == 1]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [self.input_key,]
