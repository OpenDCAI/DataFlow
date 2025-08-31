#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_and_serve.py
~~~~~~~~~~~~~~~~~~~~~
一键下载 Hugging Face 模型并用 vLLM 启动 OpenAI-API 服务。

示例：
    python download_and_serve.py \
        --repo_id meta-llama/Meta-Llama-3-8B-Instruct \
        --port 8000 \
        --dtype half \
        --quantization awq \
        --trust-remote-code
"""

import argparse
import os
import re
import subprocess
import sys
from huggingface_hub import snapshot_download


def sanitize(repo_id: str) -> str:
    """
    将 repo_id 转成安全的文件夹名，如 openai/gpt-oss-20b -> openai_gpt-oss-20b
    """
    return re.sub(r"[/@:]", "_", repo_id)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a HF model via mirror then start vLLM server."
    )

    # 下载相关
    parser.add_argument(
        "--repo_id",
        required=True,
        help="HuggingFace 仓库名，如 'openai/gpt-oss-20b'",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="镜像站地址，默认使用 HF-Mirror",
    )
    parser.add_argument(
        "--local_dir_root",
        default="./models",
        help="模型本地保存根目录",
    )

    # vLLM 常用启动参数
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口")
    parser.add_argument(
        "--dtype",
        choices=["half", "bfloat16", "float16", "float32"],
        default="half",
        help="权重精度",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        dest="gpu_mem_util",
        help="显存利用率上限",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tp_size",
        help="多卡并行推理时卡数",
    )
    parser.add_argument(
        "--quantization",
        choices=["awq", "gptq", "none"],
        default="none",
        help="是否量化权重 (awq/gptq)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="是否信任远端自定义代码",
    )

    # 其余未知参数原样透传给 vLLM
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="要直接透传给 vLLM 的其余参数，用 \"-- <args>\" 的形式放在最后",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 下载模型
    model_dir = os.path.join(args.local_dir_root, sanitize(args.repo_id))
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n[1/2] 正在下载模型 {args.repo_id} → {model_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        endpoint=args.endpoint,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("[✓] 下载完成\n")

    # 2. 组合 vLLM 启动命令
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_dir,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_mem_util),
        "--tensor-parallel-size",
        str(args.tp_size),
    ]

    if args.quantization != "none":
        cmd += ["--quantization", args.quantization]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if args.extra:
        cmd += args.extra  # 透传其它自定义参数

    print("[2/2] 启动 vLLM 服务器 ↓\n" + " ".join(cmd) + "\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()