from typing import Any, Literal

import pandas as pd
from distflow.data.types import DatasetProcessOutputItem, MessageData
from distflow.mmd import MMDDistance

from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class MMDDatasetEvaluator(OperatorABC):
    def __init__(
        self,
        ref_frame: DataFlowStorage,
        *,
        # dataset config
        ref_max_sample_num: int = 5000,
        ref_shuffle_seed: int = 42,
        ref_instruction_key: str = "input",
        ref_output_key: str = "output",
        # kernel
        kernel_type: Literal["RBF"] = "RBF",
        bias: bool = True,
        rbf_sigma: float = 1.0,
        # embedding common
        embedding_type: Literal[
            "vllm", "sentence_transformers"
        ] = "sentence_transformers",
        embedding_model_name: str | None = None,
        # sentence_transformers specific
        st_device: str = "cuda",
        st_batch_size: int = 32,
        st_normalize_embeddings: bool = True,
        # vllm specific
        vllm_max_num_seqs: int = 128,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_tensor_parallel_size: int = 1,
        vllm_pipeline_parallel_size: int = 1,
        vllm_truncate_max_length: int = 40960,
        # cache config
        cache_type: Literal["redis", "none"] = "none",
        redis_url: str = "redis://127.0.0.1:6379",
        max_concurrent_requests: int = 50,
        redis_db: int = 0,
        cache_model_id: str | None = None,
    ):
        self.logger = get_logger()
        self.logger.info(f"Initializing {self.__class__.__name__}...")

        self.ref_max_sample_num = ref_max_sample_num
        self.ref_shuffle_seed = ref_shuffle_seed
        self.ref_data = self._sample_data_helper(
            data_frame=ref_frame,
            max_sample_num=ref_max_sample_num,
            shuffle_seed=ref_shuffle_seed,
            instruction_key=ref_instruction_key,
            output_key=ref_output_key,
        )

        assert (
            embedding_model_name is not None
        ), "embedding_model_name must be specified"
        if embedding_type == "sentence_transformers":
            from distflow.embed.sentence_transformers import SentenceTransformersEmbed

            embedder = SentenceTransformersEmbed(
                model_name=embedding_model_name,
                device=st_device,
                batch_size=st_batch_size,
                normalize_embeddings=st_normalize_embeddings,
                trust_remote_code=True,
            )
        elif embedding_type == "vllm":
            from distflow.embed.vllm import VllmEmbed

            embedder = VllmEmbed(
                model_name=embedding_model_name,
                max_num_seqs=vllm_max_num_seqs,
                gpu_memory_utilization=vllm_gpu_memory_utilization,
                tensor_parallel_size=vllm_tensor_parallel_size,
                pipeline_parallel_size=vllm_pipeline_parallel_size,
                truncate_max_length=vllm_truncate_max_length,
            )
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        if cache_type == "redis":
            from distflow.cache.redis_cache import RedisCache
            from distflow.embed.cache_wrapper import CachedEmbed

            cache = RedisCache(
                redis_url=redis_url,
                max_concurrent_requests=max_concurrent_requests,
                redis_db=redis_db,
            )
            embedder = CachedEmbed(embedder, cache, cache_model_id=cache_model_id)
        elif cache_type != "none":
            raise ValueError(f"Unsupported cache_type: {cache_type}")

        self.mmd_distance = MMDDistance(
            embedder=embedder,
            kernel_type=kernel_type,
            bias=bias,
            rbf_sigma=rbf_sigma,
        )

    def _sample_data_helper(
        self,
        data_frame: DataFlowStorage,
        max_sample_num: int,
        shuffle_seed: int,
        instruction_key: str,
        output_key: str,
    ) -> list[DatasetProcessOutputItem]:
        samples: pd.DataFrame = data_frame.read("dataframe")

        if max_sample_num > 0 and max_sample_num < len(samples):
            self.logger.info(f"随机采样 {max_sample_num} 条数据")
            sampled_df = samples.sample(n=max_sample_num, random_state=shuffle_seed)
        else:
            self.logger.info("使用全部数据并打乱顺序")
            sampled_df = samples.sample(frac=1, random_state=shuffle_seed)

        sampled_df = sampled_df.reset_index(drop=True)

        instructions = sampled_df[instruction_key].to_list()
        outputs = sampled_df[output_key].to_list()
        data: list[DatasetProcessOutputItem] = []
        for instruction, output in zip(instructions, outputs):
            assert isinstance(instruction, str) and isinstance(
                output, str
            ), "Instruction and output must be strings"
            data.append(
                DatasetProcessOutputItem(
                    messages=[
                        MessageData(role="user", content=instruction),
                        MessageData(role="assistant", content=output),
                    ],
                    meta={
                        "frame": f"{data_frame!s}",
                        "instruction_key": instruction_key,
                        "output_key": output_key,
                        "max_samples": max_sample_num,
                        "shuffle_seed": shuffle_seed,
                    },
                )
            )
        return data

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "使用最大均值差异 (MMD) 方法评估两个数据集之间的分布差异。\n"
                "通过将文本嵌入到高维空间并计算核函数差异，量化评估数据集与参考数据集的分布偏移程度。\n"
                "输入参数：\n"
                "- ref_frame: 参考数据集 (DataFlowStorage)，作为分布比较的基准\n"
                "- ref_max_sample_num: 参考集最大采样数，默认 5000\n"
                "- ref_shuffle_seed: 参考集随机种子，默认 42\n"
                "- ref_instruction_key: 参考集中指令字段名，默认 'input'\n"
                "- ref_output_key: 参考集中输出字段名，默认 'output'\n"
                "- kernel_type: 核函数类型，当前仅支持 'RBF'\n"
                "- rbf_sigma: RBF 核带宽参数，默认 1.0\n"
                "- embedding_type: 嵌入模型类型，可选 'sentence_transformers' 或 'vllm'\n"
                "- embedding_model_name: 嵌入模型名称（必填）\n"
                "- st_device/st_batch_size: SentenceTransformers 设备与批次大小\n"
                "- vllm_*: vLLM 相关配置参数\n"
                "- cache_type: 缓存类型，可选 'redis' 或 'none'\n"
                "输出参数：\n"
                "- MMDScore: MMD 距离值（越小表示分布越接近）\n"
                "- MMDMeta: 包含计算细节的元数据字典"
            )
        elif lang == "en":
            return (
                "Evaluate distribution discrepancy between two datasets using Maximum Mean Discrepancy (MMD).\n"
                "Quantifies distribution shift by computing kernel-based distance between embeddings of evaluation data and reference data.\n"
                "Input Parameters:\n"
                "- ref_frame: Reference dataset (DataFlowStorage) as distribution baseline\n"
                "- ref_max_sample_num: Max samples from reference, default 5000\n"
                "- ref_shuffle_seed: Random seed for reference sampling, default 42\n"
                "- ref_instruction_key: Instruction field name in reference, default 'input'\n"
                "- ref_output_key: Output field name in reference, default 'output'\n"
                "- kernel_type: Kernel function type, currently only 'RBF' supported\n"
                "- rbf_sigma: RBF kernel bandwidth, default 1.0\n"
                "- embedding_type: Embedding backend, 'sentence_transformers' or 'vllm'\n"
                "- embedding_model_name: Embedding model name (required)\n"
                "- st_device/st_batch_size: SentenceTransformers device and batch size\n"
                "- vllm_*: vLLM configuration parameters\n"
                "- cache_type: Cache type, 'redis' or 'none'\n"
                "Output Parameters:\n"
                "- MMDScore: MMD distance value (smaller indicates closer distributions)\n"
                "- MMDMeta: Metadata dictionary with computation details"
            )
        else:
            return "Evaluate dataset distribution discrepancy using Maximum Mean Discrepancy (MMD)."

    def run(
        self,
        storage: DataFlowStorage,
        input_instruction_key: str,
        input_output_key: str,
        max_sample_num: int | None = None,
        shuffle_seed: int | None = None,
    ) -> tuple[float, dict[str, Any]]:
        max_sample_num = (
            max_sample_num if max_sample_num is not None else self.ref_max_sample_num
        )
        shuffle_seed = (
            shuffle_seed if shuffle_seed is not None else self.ref_shuffle_seed
        )
        eval_data = self._sample_data_helper(
            data_frame=storage,
            max_sample_num=max_sample_num,
            shuffle_seed=shuffle_seed,
            instruction_key=input_instruction_key,
            output_key=input_output_key,
        )
        mmd_result = self.mmd_distance.compute(
            src=eval_data,
            tgt=self.ref_data,
        )
        mmd_value = mmd_result[0].value
        mmd_meta = mmd_result[0].meta
        self.logger.info(
            f"MMDDatasetEvaluator result: MMD={mmd_value}, meta={mmd_meta}"
        )
        return mmd_value, mmd_meta
