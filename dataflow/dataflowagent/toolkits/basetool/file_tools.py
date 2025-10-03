from __future__ import annotations
import asyncio
import inspect
import sys
import os
from pydantic import BaseModel
import httpx
import json
import uuid
from typing import List, Dict, Sequence, Any, Union, Optional, Iterable, Mapping, Set, Callable
from pathlib import Path
 
from functools import lru_cache
import yaml
# from clickhouse_connect import get_client
import subprocess
from collections import defaultdict, deque
from dataflow.utils.storage import FileStorage
from dataflow import get_logger
logger = get_logger()
from dataflow.cli_funcs.paths import DataFlowPath
from dataflow.dataflowagent.storage.storage_service import SampleFileStorage
from dataflow.dataflowagent.state import DFState,DFRequest

parent_dir = f"{DataFlowPath.get_dataflow_agent_dir()}/toolkits"
MAX_JSONL_LINES = 50
DATA_DIR = Path("./data/knowledgebase")  # Local data storage directory

def local_tool_for_sample(
    state: DFRequest,
    sample_size: int = 10,
    use_file_sys: int = 1,
    cache_type: str = "jsonl",
    only_keys: bool = False,
) -> Dict[str, Any]:
    from collections import Counter
    """
    Sample, classify, and compute statistics on sample data.

    Args:
        state: Request object containing file information
        sample_size: Number of samples to retrieve.
        use_file_sys: Whether to use file system storage (1) or not (0).
        cache_type: Storage cache type ("jsonl" by default).
        only_keys: If True, return only the keys found in samples

    Returns:
        A dictionary with overall statistics and sample details.
    """
    def judge_type(sample: Dict[str, Any]) -> str:
        """
        Determine and return the type of a sample.

        Args:
            sample: The sample to be judged.

        Returns:
            The type of the sample as a string.
        """
        if not isinstance(sample, dict):
            return "Other"
        if "conversations" in sample and isinstance(sample["conversations"], list):
            ok = True
            for msg in sample["conversations"]:
                if not (
                    (isinstance(msg, dict) and "role" in msg and "content" in msg) or
                    (isinstance(msg, dict) and "from" in msg and "value" in msg)
                ):
                    ok = False
                    break
            if ok:
                return "SFT Multi-Round"
        if "instruction" in sample and "output" in sample:
            if isinstance(sample["instruction"], str) and isinstance(sample["output"], str):
                if "input" not in sample or sample["input"] is None or isinstance(sample["input"], str):
                    return "SFT Single"
        pt_keys = {"text", "content", "sentence"}
        if len(sample) == 1:
            (k, v), = sample.items()
            if k in pt_keys and isinstance(v, str):
                return "PT"
        return "Other"

    # Storage selection
    if use_file_sys:
        from dataflow.dataflowagent.storage.storage_service import SampleFileStorage
        
        # 创建存储实例
        storage = SampleFileStorage(
            first_entry_file_name=state.json_file, 
            cache_type=cache_type  # 使用传入的cache_type参数
        )
        storage.step()
        
        logger.debug(f"------------Before Sampling--------------------")
        
        # 获取总数
        total = storage.count()
        
        # 使用新的rsample方法进行采样
        samples, actual_sample_size = storage.rsample(
            mode="manual", 
            k=sample_size
        )
        
        logger.debug(f"------------After Sampling--------------------")
        logger.debug(f"Requested: {sample_size}, Actual: {actual_sample_size}, Total: {total}")
        
    else:
        # 如果不使用文件系统，返回空结果或者抛出异常
        logger.warning("Non-file system storage not implemented in new version")
        samples = []
        total = 0

    # 如果只需要keys，获取字段信息
    if only_keys:
        if use_file_sys and storage:
            # 使用新的get_fields方法
            key_set = set(storage.get_fields())
            # 如果需要从样本中获取更完整的keys
            for sample in samples:
                if isinstance(sample, dict):
                    key_set.update(sample.keys())
            return sorted(key_set)
        else:
            # 从样本中收集keys
            key_set = set().union(*(s.keys() for s in samples if isinstance(s, dict)))
            return sorted(key_set)

    # 分类样本并计算统计信息
    type_list = [judge_type(s) for s in samples]
    counter = Counter(type_list)

    # 计算分布（基于实际样本数而不是总数）
    sample_count = len(samples)
    dist = {
        t: {"count": c, "ratio": round(c / sample_count, 4) if sample_count > 0 else 0.0}
        for t, c in counter.items()
    }

    # 收集所有keys
    key_set = set().union(*(s.keys() for s in samples if isinstance(s, dict)))

    stats = {
        "total": total,
        "sample_size": sample_count,
        "stateed_size": sample_size,
        "distribution": dist,
        "samples": samples,
        "available_keys": sorted(key_set)
    }

    logger.debug(f"-------Data Statistics-------\n {stats}")
    return stats


def local_tool_for_get_categories():
    """
    返回 OPERATOR_REGISTRY 中实际注册的 operator 分类列表（如 agentic_rag, chemistry, ...）。
    """
    try:
        from dataflow.utils.registry import OPERATOR_REGISTRY
        if hasattr(OPERATOR_REGISTRY, '_init_loaders'):
            OPERATOR_REGISTRY._init_loaders()
        if hasattr(OPERATOR_REGISTRY, '_get_all'):
            OPERATOR_REGISTRY._get_all()
        categories = set()
        for name, cls in OPERATOR_REGISTRY:
            if hasattr(cls, '__module__'):
                parts = cls.__module__.split('.')
                if len(parts) >= 3 and parts[0] == 'dataflow' and parts[1] == 'operators':
                    categories.add(parts[2])
        return sorted(categories)

    except Exception as e:
        return []


if __name__ == "__main__":
    # 简单测试 local_tool_for_sample
    from dataflow.dataflowagent.state import DFRequest
    state = DFRequest(
        language="zh",
        json_file=f"{DataFlowPath.get_dataflow_dir().parent}/dataflow/example/DataflowAgent/mq_test_data.jsonl"
    )
    print(local_tool_for_sample(state,sample_size=2))
    # from dataflow.utils.registry import OPERATOR_REGISTRY
    
    # print("="*50)
    # print("OPERATOR_REGISTRY 调试")
    # print("="*50)
    
    # # 1. 查看初始状态
    # print(f"初始 _obj_map 长度: {len(OPERATOR_REGISTRY._obj_map)}")
    # print(f"初始 loader_map: {OPERATOR_REGISTRY.loader_map}")
    # print(f"初始 loader_map values: {list(OPERATOR_REGISTRY.loader_map.values())}")
    
    # # 2. 手动触发加载器初始化
    # print("\n手动触发 _init_loaders()...")
    # try:
    #     OPERATOR_REGISTRY._init_loaders()
    #     print("✓ _init_loaders() 执行成功")
    #     print(f"加载后 loader_map values: {[type(v).__name__ for v in OPERATOR_REGISTRY.loader_map.values()]}")
    # except Exception as e:
    #     print(f"✗ _init_loaders() 失败: {e}")
    
    # # 3. 手动触发加载所有操作符
    # print("\n手动触发 _get_all()...")
    # try:
    #     OPERATOR_REGISTRY._get_all()
    #     print("✓ _get_all() 执行成功")
    #     print(f"_get_all() 后 _obj_map 长度: {len(OPERATOR_REGISTRY._obj_map)}")
    # except Exception as e:
    #     print(f"✗ _get_all() 失败: {e}")
    #     # 如果 _get_all() 失败，尝试手动调用每个loader的 _import_all()
    #     print("尝试手动加载每个模块...")
    #     for module_name, loader in OPERATOR_REGISTRY.loader_map.items():
    #         if loader is not None:
    #             try:
    #                 print(f"  加载 {module_name}...")
    #                 loader._import_all()
    #                 print(f"    ✓ {module_name} 加载成功")
    #             except Exception as me:
    #                 print(f"    ✗ {module_name} 加载失败: {me}")
    #     print(f"手动加载后 _obj_map 长度: {len(OPERATOR_REGISTRY._obj_map)}")
    
    # # 4. 创建 _NAME2CLS 并显示结果
    # _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}
    
    # print(f"\n最终结果:")
    # print(f"_NAME2CLS 长度: {len(_NAME2CLS)}")
    # print(f"_NAME2CLS keys (前20个): {list(_NAME2CLS.keys())[:20]}")
    
    # if _NAME2CLS:
    #     print(f"\n前10个操作符详情:")
    #     for i, (name, cls) in enumerate(_NAME2CLS.items()):
    #         if i >= 10:
    #             break
    #         print(f"  {i+1}. {name}: {cls}")
    #         if hasattr(cls, '__module__'):
    #             print(f"     模块: {cls.__module__}")
    # else:
    #     print("⚠️ _NAME2CLS 仍然为空!")
        
    #     # 如果还是空的，尝试直接调用 get() 方法来触发单个操作符的加载
    #     print("\n尝试使用 get() 方法触发加载...")
    #     # 随便试一个可能的操作符名称
    #     test_names = ["TextOperator", "FileReader", "DataProcessor", "BaseOperator"]
    #     for test_name in test_names:
    #         try:
    #             cls = OPERATOR_REGISTRY.get(test_name)
    #             print(f"✓ 成功获取 {test_name}: {cls}")
    #             break
    #         except KeyError:
    #             print(f"✗ {test_name} 不存在")
    #         except Exception as e:
    #             print(f"✗ 获取 {test_name} 时出错: {e}")
        
    #     # 再次检查
    #     _NAME2CLS = {name: cls for name, cls in OPERATOR_REGISTRY}
    #     print(f"使用 get() 后 _NAME2CLS 长度: {len(_NAME2CLS)}")
    
    # # 5. 分析分类
    # if _NAME2CLS:
    #     print(f"\n按模块分类:")
    #     module_counts = {}
    #     for name, cls in _NAME2CLS.items():
    #         if hasattr(cls, '__module__'):
    #             module_parts = cls.__module__.split('.')
    #             if len(module_parts) > 2 and module_parts[0] == 'dataflow' and module_parts[1] == 'operators':
    #                 category = module_parts[2]  # 取 dataflow.operators.xxx 中的 xxx
    #             else:
    #                 category = cls.__module__
                
    #             module_counts[category] = module_counts.get(category, 0) + 1
        
    #     for category, count in sorted(module_counts.items()):
    #         print(f"  {category}: {count} 个操作符")
    
    # print("="*50)