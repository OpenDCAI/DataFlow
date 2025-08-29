#!/usr/bin/env python3
"""
post_processor.py ── Post-processing Utilities for Operator Code and Pipeline Results
Author  : [Zhou Liu]
License : MIT
Created : 2024-07-10

This module provides utility functions for handling post-processing tasks in model-driven or agent-based
dataflow systems, especially those involving operator code generation and pipeline structure extraction.

Features:
* Operator code saver: Safely writes generated operator code to disk, creating necessary directories.
* Pipeline result combiner: Merges and formats pipeline edge/node information from multi-stage task outputs,
  normalizes node IDs, and prepares a standardized result for downstream consumption or visualization.
* Flexible handling of different node info formats (list of dicts or dict of operator descriptions).
* Robust against malformed or incomplete input data, with normalization and filtering logic for nodes and edges.

Designed for use in automatic code generation, LLM agent pipelines, AutoML flows, or similar systems
where dynamic assembly and export of operator code and pipeline graphs are needed.

Thread-safety: This module is not inherently thread-safe. If used in parallel processing contexts,
appropriate synchronization or file locking is recommended.

Dependencies:
- Python 3.8+
- Standard library: pathlib, uuid, typing, json
- Project-local utilities: get_operator_content, get_operator_descriptions, ChatAgentRequest

Typical Usage:
    from pipeline_postprocess import post_process_save_op_code, post_process_combine_pipeline_result

    # Save generated operator code
    post_process_save_op_code(request, last_result)

    # Combine and format pipeline results after multi-stage processing
    pipeline_result = post_process_combine_pipeline_result(last_result, task_results)

"""
from pathlib import Path
import uuid
from typing import Dict, Any, List, Union
import json
from .tools import get_operator_content,get_operator_descriptions
from .tools import ChatAgentRequest
 
def post_process_save_op_code(
    request: ChatAgentRequest,
    last_result: Dict[str, Any],
    **kwargs
) -> Dict[str, Union[str, int]]:
    try:
        code: str = last_result["code"]
        py_path = Path(request.py_path).expanduser().resolve()
        py_path.parent.mkdir(parents=True, exist_ok=True)
        bytes_written = py_path.write_text(code, encoding="utf-8")

        return {
            "status": "success",
            "path": str(py_path),
            "size": bytes_written,
        }
    except KeyError:
        return {
            "status": "error",
            "message": "The `last_result` dictionary is missing the 'code' key; unable to save the code."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save code: {repr(e)}"
        }

def post_process_combine_pipeline_result(
        last_result: Dict[str, Any],
        task_results: List,
        **kwargs
) -> Dict[str, Any]:
    edges = last_result.get("edges")
    if not isinstance(edges, list):
        return last_result
# get_operator_content
    nodes_info: Union[List[Dict[str, Any]], Dict[str, str], None] = None
    for past in reversed(task_results):
        if "ContentSubType" in past:
                nodes_info = get_operator_content(request= "", data_key= past , keep_keys= None)    
    print("nodes_info:",nodes_info)
    # Handle case when nodes_info is None or not a list/dict
    if nodes_info is None:
        return last_result
    id_map: Dict[str, Dict[str, Any]] = {}
    # Case 1: nodes_info is a dictionary of operator descriptions (for MIXTURE)
    if isinstance(nodes_info, dict):
        for node_id, (op_name, description) in enumerate(nodes_info.items(), start=1):
            key = f"node{node_id}"
            id_map[key] = {
                "node": node_id,
                "name": op_name,
                "description": description
            }
    # Case 2: nodes_info is a list of node dictionaries
    elif isinstance(nodes_info, list):
        for node_dict in nodes_info:
            if not isinstance(node_dict, dict):
                continue
            node_id = node_dict.get("node")
            if isinstance(node_id, int):
                key = f"node{node_id}"
                id_map[key] = node_dict

    def normalize(nid: Any) -> Union[str, None]:
        if isinstance(nid, int):
            return f"node{nid}"
        if isinstance(nid, str):
            return nid if nid.startswith("node") else f"node{nid}"
        return None

    seen_ids = set()
    out_nodes: List[Dict[str, Any]] = []
    normalized_edges: List[Dict[str, str]] = []

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = normalize(edge.get("source"))
        tgt = normalize(edge.get("target"))
        if src is None or tgt is None:
            continue
        normalized_edges.append({"source": src, "target": tgt})
        for rid in (src, tgt):
            node_dict = id_map.get(rid)
            if not node_dict:
                continue
            node_id = node_dict["node"]
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            filtered = {
                k: v
                for k, v in node_dict.items()
                if k not in ("input", "output", "node")
            }
            filtered["id"] = rid
            out_nodes.append(filtered)

    new_result = dict(last_result)
    new_result["nodes"] = out_nodes
    new_result["edges"] = normalized_edges
    new_result["name"] = f"{uuid.uuid4()}_pipeline"
    return new_result

if __name__ == "__main__":
    pass