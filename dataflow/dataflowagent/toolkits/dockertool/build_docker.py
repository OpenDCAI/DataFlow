"""
Python 版镜像构建脚本：
1. 写 requirements.txt / Dockerfile
2. docker build
3. (可选) docker push
"""

import argparse
import subprocess
from pathlib import Path
import textwrap
import sys

DEFAULT_REQS = """\
pandas==2.2.2
numpy==1.26.4
sqlalchemy>=2.0,<3.0




"""

DOCKERFILE_TMPL = """\
FROM python:{py_version}-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential gcc \\
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN useradd -ms /bin/bash appuser
USER appuser
WORKDIR /app
ENTRYPOINT ["python"]
"""

def run(cmd: list[str], **kw):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="myorg/dataflow-py:3.11",
                        help="镜像名:TAG")
    parser.add_argument("--py", "--python", dest="py_version",
                        default="3.11", help="Python 主版本")
    parser.add_argument("--push", action="store_true",
                        help="构建后 docker push")
    opts = parser.parse_args()

    root = Path(__file__).resolve().parent
    docker_dir = root / "docker"
    docker_dir.mkdir(exist_ok=True)

    # 1. requirements.txt
    (docker_dir / "requirements.txt").write_text(DEFAULT_REQS, encoding="utf-8")
    print("✓ requirements.txt written")

    # 2. Dockerfile
    dockerfile_str = DOCKERFILE_TMPL.format(py_version=opts.py_version)
    (docker_dir / "Dockerfile").write_text(dockerfile_str, encoding="utf-8")
    print("✓ Dockerfile written")

    # 3. build
    run(["docker", "build", "-t", opts.image, str(docker_dir)])

    # 4. push
    if opts.push:
        run(["docker", "push", opts.image])

    print(f"\n 完成！请把 DOCKER_IMAGE 改为 '{opts.image}'\n")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)