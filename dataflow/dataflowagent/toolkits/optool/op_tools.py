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

import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger

log = get_logger()
RESOURCE_DIR = Path(__file__).resolve().parent.parent / "resources"
OPS_JSON_PATH = RESOURCE_DIR / "ops.json"

def local_tool_for_get_purpose(req: DFRequest) -> str:
    return req.target or ""

# ===================================================================更新算子库部分代码：
# 工具函数：安全调用带 @staticmethod 的 get_desc(lang)
def _call_get_desc_static(cls, lang: str = "zh") -> str | None:
    """
    仅当类的 get_desc 被显式声明为 @staticmethod 时才调用。
    兼容两种签名: (lang) 或 (self, lang)。
    返回 None 表示跳过此算子。
    """
    func_obj = cls.__dict__.get("get_desc")
    if not isinstance(func_obj, staticmethod):
        return None

    fn = func_obj.__func__
    params = list(inspect.signature(fn).parameters)
    try:
        if params == ["lang"]:
            return fn(lang)
        if params == ["self", "lang"]:
            return fn(None, lang)
    except Exception as e:
        log.warning(f"调用 {cls.__name__}.get_desc 失败: {e}")
    return None


# ---------------------------------------------------------------------------
def _param_to_dict(p: inspect.Parameter) -> Dict[str, Any]:
    """把 inspect.Parameter 转成 JSON 可序列化的字典（参考 MCP func 定义）"""
    return {
        "name": p.name,
        "default": None if p.default is inspect.Parameter.empty else p.default,
        "kind": p.kind.name,  # POSITIONAL_OR_KEYWORD / VAR_POSITIONAL / ...
    }


def _get_method_params(
    method: Any, skip_first_self: bool = False
) -> List[Dict[str, Any]]:
    """
    提取方法形参，转换为列表。
    skip_first_self=True 时会丢掉第一个 self 参数。
    """
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.values())
        if skip_first_self and params and params[0].name == "self":
            params = params[1:]
        return [_param_to_dict(p) for p in params]
    except Exception as e:
        log.warning(f"获取方法参数出错: {e}")
        return []


def _gather_single_operator(
    op_name: str, cls: type, node_index: int
) -> Tuple[str, Dict[str, Any]]:
    """
    收集单个算子的全部信息，返回 (category, info_dict)
    """
    # 1) 分类：dataflow.operators.<category>.xxx
    category = "unknown"
    if hasattr(cls, "__module__"):
        parts = cls.__module__.split(".")
        if len(parts) >= 3 and parts[0] == "dataflow" and parts[1] == "operators":
            category = parts[2]

    # 2) 描述
    description = _call_get_desc_static(cls, lang="zh") or ""

    # 3) command 形参
    init_params = _get_method_params(cls.__init__, skip_first_self=True)
    run_params = _get_method_params(getattr(cls, "run", None), skip_first_self=True)

    info = {
        "node": node_index,
        "name": op_name,
        "description": description,
        "parameter": {
            "init": init_params,
            "run": run_params,
        },
        # 下面三项暂时留空，后续有需要再填
        "required": "",
        "depends_on": [],
        "mode": "",
    }
    return category, info


def _dump_all_ops_to_file() -> Dict[str, List[Dict[str, Any]]]:
    """
    遍历 OPERATOR_REGISTRY，构建完整字典并写入 ops.json。
    额外添加 "Default" → 所有算子全集。
    """
    log.info("开始扫描 OPERATOR_REGISTRY，生成 ops.json ...")

    if hasattr(OPERATOR_REGISTRY, "_init_loaders"):
        OPERATOR_REGISTRY._init_loaders()
    if hasattr(OPERATOR_REGISTRY, "_get_all"):
        OPERATOR_REGISTRY._get_all()

    all_ops: Dict[str, List[Dict[str, Any]]] = {}
    default_bucket: List[Dict[str, Any]] = []

    idx = 1
    for op_name, cls in OPERATOR_REGISTRY:
        category, info = _gather_single_operator(op_name, cls, idx)
        all_ops.setdefault(category, []).append(info)   
        default_bucket.append(info)                     # 收集全集
        idx += 1

    all_ops["Default"] = default_bucket

    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(OPS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(all_ops, f, ensure_ascii=False, indent=2)
        log.info(f"算子信息已写入 {OPS_JSON_PATH}")
    except Exception as e:
        log.warning(f"写入 {OPS_JSON_PATH} 失败: {e}")

    return all_ops

def _ensure_ops_cache() -> Dict[str, List[Dict[str, Any]]]:
    """
    若 ops.json 不存在或为空，则重新生成。
    返回文件中的全部数据。
    """
    if OPS_JSON_PATH.exists():
        try:
            with open(OPS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:  # 非空文件
                return data
        except Exception as e:
            log.warning(f"读取 {OPS_JSON_PATH} 失败，将重新生成: {e}")
    return _dump_all_ops_to_file()


# 供 LangChain Tool 调用的主函数
def get_operator_content(data_type: str) -> str:
    """
    根据传入的 `data_type`（即算子类别，如 "text2sql", "rag" …）
    返回该类别下所有算子的 JSON 字符串。

    如果该类别不存在，返回 "[]"
    """
    # all_ops = _ensure_ops_cache()
    all_ops = _dump_all_ops_to_file()

    import copy

    if data_type in all_ops:
        content = copy.deepcopy(all_ops[data_type])
    else:
        content = []

    # 作为字符串返回，方便 LLM 直接嵌入提示词
    return json.dumps(content, ensure_ascii=False, indent=2)


def get_operator_content_str(data_type: str) -> str:
    """
    返回该类别下所有算子的 “name:描述” 长字符串，用分号分隔。
    """
    all_ops = _dump_all_ops_to_file()  # 或 _ensure_ops_cache()
    raw_items = all_ops.get(data_type, [])

    # 用英文引号，如果有需要可用中文引号
    lines = [
        f'"{item.get("name", "")}":"{item.get("description", "")}"'
        for item in raw_items
    ]

    return "; ".join(lines)


def post_process_combine_pipeline_result(results: Dict) -> str:

    return "hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"


# if __name__ == "__main__":
#     print(get_operator_content("text2sql"))


# =================================================================== 算子RAG部分代码：
import os, httpx, numpy as np, faiss
from typing import List

def _call_openai_embedding_api(
    texts: List[str],
    model_name: str = "text-embedding-ada-002",
    base_url: str = "https://api.openai.com/v1/embeddings",
    api_key: str | None = None,
    timeout: float = 30.0,
) -> np.ndarray:
    """
    入参:
        texts      : 待编码的字符串列表
        model_name : OpenAI embedding model，如 'text-embedding-ada-002'
        base_url   : 你的服务器地址，默认直连官方
        api_key    : OpenAI API key，若为 None 则读取环境变量 OPENAI_API_KEY
    返回:
        shape=(len(texts), dim) 的 np.ndarray(float32)，已做 L2 归一化
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("必须提供 OpenAI API-Key，可通过参数或环境变量 OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    vecs: List[List[float]] = []
    with httpx.Client(timeout=timeout) as client:
        for t in texts:
            resp = client.post(
                base_url,
                headers=headers,
                json={"model": model_name, "input": t},
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"调用 OpenAI embedding 失败: {e}\n{resp.text}"
                ) from e

            try:
                data = resp.json()
                # OpenAI 返回格式: {"data":[{"embedding":[...], "index":0, ...}]}
                vec = data["data"][0]["embedding"]
            except Exception as e:
                raise RuntimeError(f"解析返回 JSON 失败: {resp.text}") from e

            vecs.append(vec)

    arr = np.asarray(vecs, dtype=np.float32)
    # L2 归一化，方便后续用内积≈余弦相似度
    faiss.normalize_L2(arr)
    return arr

def rag_build_index(
    category: str,
    ops_json_path: str,
    model_name: str = "text-embedding-ada-002",
    base_url: str = "https://api.openai.com/v1/embeddings",
    api_key: str | None = None,
    top_k: int = 5,
):
    """
    示例构建向量索引，并返回 Top-K 结果验证
    """
    # 1) 载入指定类别的算子文本
    with open(ops_json_path, "r", encoding="utf-8") as f:
        all_ops = json.load(f)

    ops = all_ops.get(category, [])
    texts = [f"{o['name']} {o.get('description','')}" for o in ops]

    # 2) 调用 OpenAI 取向量
    embeddings = _call_openai_embedding_api(
        texts, model_name=model_name, base_url=base_url, api_key=api_key
    )

    # 3) 建立简单的 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)          
    index.add(embeddings)

    # 4) 写一个内部查询函数
    def _search(query: str, k: int = top_k) -> List[str]:
        q_vec = _call_openai_embedding_api(
            [query], model_name=model_name, base_url=base_url, api_key=api_key
        )
        D, I = index.search(q_vec, k)      
        return [ops[i]["name"] for i in I[0]]

    # 5) 返回查询函数或立刻测试
    return _search

def get_operators_by_rag(search_goal: str,category: str = "text2sql",top_k: int = 4) -> str:
    OPS_JSON = "/mnt/h_h_public/lh/lz/DataFlow/dataflow/dataflowagent/toolkits/resources/ops.json"

    search_fn = rag_build_index(
        category="text2sql",
        ops_json_path=OPS_JSON,
        model_name="text-embedding-3-small",
        base_url="http://123.129.219.111:3000/v1/embeddings",
        api_key="sk-",
        top_k=4,
    )
    return search_fn(search_goal)

if __name__ == "__main__":
    print(get_operators_by_rag("将自然语言转换为SQL查询语句"))

