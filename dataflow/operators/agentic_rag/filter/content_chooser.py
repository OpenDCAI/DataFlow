import numpy as np
import pandas as pd
import random
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
        embedding (Tensor): Embedding vector extracted from a CNN
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

        if cluster_centers:
            centers = self.features[cluster_centers]

            distance = F.pairwise_distance(self.features, centers, p=2).reshape(-1, 1)

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
          indices of samples selected to minimize distance to cluster centers
        """

        if selected_idxs is None:
            selected_idxs = []

        if self.embedding.ndim == 2:
            # self.model.fit(self.embedding)
            # self.features = self.model.transform(self.embedding)
            
            self.features = self.embedding
            self.reset_distances()
        else:
            self.features = self.embedding.reshape(self.embedding.shape[0], -1)
            self.update_distances(cluster_centers=selected_idxs)

        selected_coreset_idxs: List[int] = []
        idx = int(torch.randint(high=self.n_observations, size=(1,)).item())
        cnt = 0
        for _ in range(self.coreset_size):
            cnt += 1
            if(cnt % 1000 == 0):
                print(cnt)
            self.update_distances(cluster_centers=[idx])
            idx = self.get_new_idx()
            if idx in selected_idxs:
                raise ValueError("New indices should not be in selected indices.")
            self.min_distances[idx] = 0
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
class ContentChooser(OperatorABC):
    def __init__(self, num_samples: int, method: str = "random", embedding_serving : LLMServingABC = None):
        self.num_samples = num_samples
        self.method = method
        self.embedding_serving = embedding_serving
        self.logger = get_logger()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "该算子用于从大量的文档片段中选取部分文档片段，用于后续生成种子QA对\n\n"
                "输入参数:\n"
                "- input_key: 包含文档片段的字段名\n"
                "- embedding_model_path: 嵌入模型路径\n"
                "- num_samples: 选取的文档片段数量\n"
                "- method: 选择方法，随机或k-center-greedy\n\n"
            )
        elif lang == "en":
            return (
                "This operator chooses document fragments for seed QA pairs.\n\n"
                "Input Parameters:\n"
                "- input_key: Field name containing the content\n"
                "- embedding_serving: Embedding serving\n"
                "- num_samples: Number of document fragments to select\n"
                "- method: Selection method, random or k-center-greedy\n\n"
                "Output Parameters:\n"
                "- Returns 1 for valid content, 0 otherwise"
            )
        else:
            return "ContentChooser chooses document fragments for seed QA pairs"
    
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
        indexes =  np.zeros(len(dataframe)).astype(int)
        
        if self.method == "random":
            chooss_indexes = random.sample(range(len(texts)), self.num_samples)
        elif self.method == "kcenter":
            embeddings_list = self.embedding_serving.generate_embedding_from_input(texts)
            embeddings = torch.tensor(embeddings_list)

            sampler = KCenterGreedy(embedding=embeddings, sampling_ratio= self.num_samples / len(texts))
            chooss_indexes = sampler.select_coreset_idxs()
        else:
            raise ValueError(f"Invalid method: {self.method}")

        for index in chooss_indexes:
            indexes[index] = 1

        dataframe = dataframe[np.array(indexes) == 1]

        output_file = storage.write(dataframe)
        self.logger.info(f"Results saved to {output_file}")
        
        return [self.input_key,]