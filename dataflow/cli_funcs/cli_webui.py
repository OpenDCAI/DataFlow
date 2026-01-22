from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import webbrowser
import requests

from dataflow.cli_funcs.utils import _echo

OWNER = "OpenDCAI"
REPO = "DataFlow-WebUI"
LATEST_API = f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest"
REPO_URL = f"https://github.com/{OWNER}/{REPO}"


def _ask_yes(prompt: str, default_no: bool = True) -> bool:
    suffix = " [y/N]: " if default_no else " [Y/n]: "
    ans = input(prompt + suffix).strip().lower()
    if not ans:
        return not default_no
    return ans in {"y", "yes"}


def _confirm_yes() -> None:
    _echo(f"This will run DataFlow-WebUI ({REPO_URL}) by using a GitHub Release zip.", "yellow")
    if input("Type 'yes' to continue: ").strip().lower() != "yes":
        raise SystemExit(0)


def _pick_zip(release: dict) -> Tuple[str, str, str]:
    tag = release.get("tag_name") or "latest"
    for a in release.get("assets", []):
        name = a.get("name", "")
        url = a.get("browser_download_url", "")
        if name.startswith("DataFlow-WebUI-") and name.endswith(".zip") and url:
            return tag, name, url
    raise RuntimeError("No DataFlow-WebUI-*.zip found in latest release.")


def _ask_base_dir(default: Path) -> Path:
    _echo(f"Default download directory: {default}", "cyan")
    ans = input(f"Download directory [{default}]: ").strip()
    return (Path(ans).expanduser().resolve() if ans else default.resolve())

def _open_browser(url: str) -> None:
    try:
        ok = webbrowser.open(url, new=2)  # new=2: new tab if possible
        if ok:
            _echo(f"Opened browser: {url}", "green")
        else:
            _echo(f"Please open in browser: {url}", "cyan")
    except Exception:
        _echo(f"Please open in browser: {url}", "cyan")

def _download_with_progress(url: str, dst: Path) -> None:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            last_percent = -1

            with open(dst, "wb") as f:
                for chunk in r.iter_content(chunk_size=100 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total > 0:
                        percent = int(downloaded * 100 / total)
                        if percent != last_percent:
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total / (1024 * 1024)
                            print(
                                f"\rDownloading: {percent:3d}% "
                                f"({mb_done:.1f}/{mb_total:.1f} MB)",
                                end="",
                                flush=True,
                            )
                            last_percent = percent

            if total > 0:
                print()  # 换行
    except Exception as e:
        print(e)
        if dst.exists():
            dst.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed: {e}, please mannually fetch it from {url}") from e


def cli_webui(
    zippath: Optional[Path] = None,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    _confirm_yes()

    # 1) 选择 base dir（默认 cwd/dataflow_webui）
    base_dir = _ask_base_dir(Path.cwd() / "dataflow_webui")
    downloads = base_dir / "downloads"
    releases = base_dir / "releases"
    downloads.mkdir(parents=True, exist_ok=True)
    releases.mkdir(parents=True, exist_ok=True)

    _echo(f"Base directory: {base_dir}", "green")

    # 2) 确定 zip（本地 or 最新 release）
    if zippath:
        zip_path = Path(zippath).expanduser().resolve()
        if not zip_path.is_file():
            raise RuntimeError(f"zippath not found: {zip_path}")
        tag = "local"
        _echo(f"Using local zip: {zip_path}", "green")
    else:
        r = requests.get(LATEST_API, headers={"Accept": "application/vnd.github+json"}, timeout=20)
        r.raise_for_status()
        release = r.json()
        tag, name, url = _pick_zip(release)
        zip_path = downloads / name

        # 2.1 先检测是否存在，再决定要不要下载
        if zip_path.exists() and zip_path.stat().st_size > 0:
            _echo(f"Found existing zip: {zip_path}", "yellow")
            if _ask_yes("Overwrite and re-download this zip?", default_no=True):
                _echo(f"Re-downloading → {zip_path}", "cyan")
                zip_path.unlink(missing_ok=True)
                _echo(f"Downloading: {name}", "cyan")
                _download_with_progress(url, zip_path)

            else:
                _echo("Using existing zip (skip download).", "green")
        else:
            _echo(f"Will download: {name}", "cyan")
            _echo(f"Download to : {zip_path}", "cyan")
            with requests.get(url, stream=True, timeout=120) as rr:
                rr.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in rr.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)

    # 3) 解压（同样：先检测是否存在，再问是否覆盖）
    extract_dir = releases / tag
    if extract_dir.exists():
        _echo(f"Found existing extracted dir: {extract_dir}", "yellow")
        if _ask_yes("Overwrite and re-extract?", default_no=True):
            shutil.rmtree(extract_dir)
            _echo(f"Extracting → {extract_dir}", "cyan")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        else:
            _echo("Using existing extracted files (skip extract).", "green")
    else:
        _echo(f"Extracting → {extract_dir}", "cyan")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # 4) 定位 backend
    dirs = [p for p in extract_dir.iterdir() if p.is_dir()]
    root = dirs[0] if len(dirs) == 1 else extract_dir
    backend = root / "backend"
    if not backend.exists():
        raise RuntimeError("backend/ directory not found after extraction.")

    _echo(f"Backend directory: {backend}", "green")

    # 5) 安装依赖（当前环境） + 运行
    _echo("Installing backend requirements into current Python environment...", "cyan")
    os.system(f"cd '{backend}' && python -m pip install -r requirements.txt")

    _echo(f"Starting WebUI at http://{host}:{port}/ui/", "green")
    _open_browser(f"http://{host}:{port}/ui/")
    os.system(
        f"cd '{backend}' && "
        f"python -m uvicorn app.main:app "
        f"--reload --reload-dir app "
        f"--host {host} --port {port}"
    )
