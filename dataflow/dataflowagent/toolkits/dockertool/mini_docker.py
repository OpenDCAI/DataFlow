from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import docker                       # pip install docker
from docker.errors import DockerException, NotFound

DOCKER_IMAGE = "python:3.11-slim"   # 可按需切换
RUN_TIMEOUT  = 120                  # s

def _run_in_container_sync(
    file_path: Path,
    image: str = DOCKER_IMAGE,
    timeout: int = RUN_TIMEOUT,
) -> Dict[str, Any]:
    """
    同步：在隔离 Docker 容器中执行脚本，返回结果字典。
    """
    client = docker.from_env()

    # 确保镜像已就绪
    try:
        client.images.get(image)
    except NotFound:
        for line in client.api.pull(image, stream=True, decode=True):
            # 可在此处打印进度；为保持简洁这里只拉取
            pass

    container = None
    try:
        # 以 bind-mount 方式把脚本挂到 /app/script.py（只读）
        container = client.containers.run(
            image=image,
            command=["python", "/app/script.py"],
            volumes={
                str(file_path): {
                    "bind": "/app/script.py",
                    "mode": "ro",
                }
            },
            detach=True,
            stdout=True,
            stderr=True,
            # ---------- 安全 / 资源限制 ----------
            network_disabled=True,
            read_only=True,
            mem_limit="512m",
            pids_limit=128,
            cpu_quota=100000,  # 100 ms/100 ms ≈ 1 CPU
        )

        exit_status = container.wait(timeout=timeout)
        return_code = exit_status.get("StatusCode", 137)  # 137=Killed

        stdout_bytes = container.logs(stdout=True, stderr=False)
        stderr_bytes = container.logs(stdout=False, stderr=True)

        result = {
            "success": return_code == 0,
            "return_code": return_code,
            "stdout": stdout_bytes.decode(),
            "stderr": stderr_bytes.decode(),
            "file_path": str(file_path),
        }
    except DockerException as e:
        result = {
            "success": False,
            "return_code": -1,
            "stdout": "",
            "stderr": f"Docker error: {e}",
            "file_path": str(file_path),
        }
    finally:
        if container:
            # 清理容器，避免残留
            try:
                container.remove(force=True)
            except Exception:
                pass
        client.close()

    return result


async def _run_py_in_docker(
    file_path: Path,
    image: str = DOCKER_IMAGE,
    timeout: int = RUN_TIMEOUT,
) -> Dict[str, Any]:
    """
    异步封装：在事件循环中调用同步 Docker 逻辑。
    """
    return await asyncio.to_thread(_run_in_container_sync, file_path, image, timeout)