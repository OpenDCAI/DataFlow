import os
import time
import json
import requests
from typing import List, Dict
import zipfile

def _safe_basename(path_or_name: str) -> str:
    base = os.path.basename(path_or_name)
    if base.lower().endswith(".pdf"):
        base = base[:-4]
    # keep it filesystem-safe
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in base)

def _zip_name(item: dict) -> str:
    data_id = item.get("data_id", "unknown")
    name = _safe_basename(item.get("file_name", f"file_{data_id}"))
    return f"{data_id}_{name}.zip"

def _unzip_and_get_md(zip_path: str, extract_dir: str) -> str:
    """
    Unzip zip_path into extract_dir and return extract_dir/full.md
    """
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    md_path = os.path.join(extract_dir, "full.md")
    if not os.path.exists(md_path):
        raise RuntimeError(f"Expected markdown not found: {md_path}")

    return md_path


class MinerUBatchExtractorViaAPI:
    def __init__(
        self,
        api_key: str,
        model_version: str = "vlm",
        poll_interval: int = 5,
        timeout: int = 600,
    ):
        self.api_key = api_key
        self.model_version = model_version
        self.poll_interval = poll_interval
        self.timeout = timeout

        self.base_url = "https://mineru.net/api/v4"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ---------- Step 1: apply upload urls ----------
    def apply_upload_urls(self, file_paths: List[str]) -> Dict:
        files = [
            {"name": path, "data_id": str(i)}
            for i, path in enumerate(file_paths)
        ]

        payload = {
            "files": files,
            "model_version": self.model_version,
        }

        resp = requests.post(
            f"{self.base_url}/file-urls/batch",
            headers=self.headers,
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        if data["code"] != 0:
            raise RuntimeError(f"Apply upload urls failed: {data}")

        return data["data"]  # batch_id + file_urls

    # ---------- Step 2: upload files ----------
    def upload_files(self, file_paths: List[str], file_urls: List[str]):
        assert len(file_paths) == len(file_urls)

        for path, url in zip(file_paths, file_urls):
            with open(path, "rb") as f:
                r = requests.put(url, data=f)
                if r.status_code != 200:
                    raise RuntimeError(f"Upload failed for {path}")

    # ---------- Step 3: poll extraction result ----------
    def poll_results(self, batch_id: str) -> dict:
        start = time.time()
        url = f"{self.base_url}/extract-results/batch/{batch_id}"

        while True:
            resp = requests.get(url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()

            if data["code"] != 0:
                raise RuntimeError(f"Polling failed: {data}")

            result_list = data["data"].get("extract_result", [])

            if not result_list:
                raise RuntimeError("No extract_result returned")

            states = [item["state"] for item in result_list]

            # ---- derive batch status ----
            if all(s == "done" for s in states):
                return data["data"]

            if any(s == "failed" for s in states):
                failed = [i for i in result_list if i["state"] == "failed"]
                raise RuntimeError(f"Extraction failed: {failed}")

            # still running
            if time.time() - start > self.timeout:
                raise TimeoutError("Polling timed out")

            time.sleep(self.poll_interval)


    # ---------- Step 4: download results ---------
    def download_results(self, extract_result: list, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)

        md_results = {}

        for item in extract_result:
            if item.get("state") != "done":
                continue

            zip_url = item.get("full_zip_url")
            if not zip_url:
                raise RuntimeError(f"Missing full_zip_url for item: {item}")

            data_id = item.get("data_id", "unknown")

            # 解压目录固定为 ./<data_id>
            extract_dir = os.path.join(out_dir, str(data_id))
            zip_path = os.path.join(out_dir, f"{data_id}.zip")

            # 1. download zip
            r = requests.get(zip_url, timeout=120)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)

            print(f"[OK] Downloaded ZIP: {zip_path}")

            # 2. unzip and get full.md
            md_path = _unzip_and_get_md(zip_path, extract_dir)

            print(f"[OK] Markdown ready: {md_path}")

            md_results[data_id] = {
                "zip_path": zip_path,
                "extract_dir": extract_dir,
                "md_path": md_path,
            }

        return md_results



    # ---------- One-call pipeline ----------
    def extract_batch(self, file_paths: list, out_dir: str) -> dict:
        apply_data = self.apply_upload_urls(file_paths)

        batch_id = apply_data["batch_id"]
        file_urls = apply_data["file_urls"]

        self.upload_files(file_paths, file_urls)

        result_data = self.poll_results(batch_id)

        # 关键：接住 download_results 的返回值
        md_results = self.download_results(
            result_data["extract_result"],
            out_dir
        )

        items = []
        for it in result_data.get("extract_result", []):
            data_id = it.get("data_id")

            info = md_results.get(data_id, {})

            items.append({
                "data_id": data_id,
                "file_name": it.get("file_name"),
                "state": it.get("state"),

                # 来自 download_results
                "zip_path": info.get("zip_path"),
                "extract_dir": info.get("extract_dir"),
                "md_path": info.get("md_path"),
            })

        return {
            "batch_id": batch_id,
            "output_dir": out_dir,
            "num_files": len(file_paths),
            "items": items,
        }
