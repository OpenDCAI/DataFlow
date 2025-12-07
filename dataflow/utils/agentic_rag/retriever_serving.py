from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from flashrag.config import Config
from flashrag.utils import get_retriever
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import uvicorn

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    topk: int = 5

retriever_list = []
available_retrievers = deque()
retriever_semaphore = None
retriever_lock = None

def init_retrievers(config_path: str, num_retriever: int):
    global retriever_list, available_retrievers, retriever_semaphore, retriever_lock
    config = Config(config_path)
    for i in range(num_retriever):
        print(f"Initializing retriever {i+1}/{num_retriever}")
        retriever = get_retriever(config)
        retriever_list.append(retriever)
        available_retrievers.append(i)
    retriever_semaphore = asyncio.Semaphore(num_retriever)
    retriever_lock = asyncio.Lock()

@app.post("/retrieve")
async def retrieve_docs(request_data: QueryRequest):
    query = request_data.query
    topk = request_data.topk

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async with retriever_semaphore:
        async with retriever_lock:
            if not available_retrievers:
                raise HTTPException(status_code=503, detail="No retriever available at the moment.")
            retriever_idx = available_retrievers.popleft()

        try:
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                results, scores = await loop.run_in_executor(
                    executor,
                    retriever_list[retriever_idx].batch_search,
                    [query], topk, True
                )

            if not results or not results[0]:
                return {"results": [], "message": "No results found."}

            docs = []
            for doc, score in zip(results[0], scores[0]):
                docs.append({
                    "doc_id": doc.get("id", "N/A"),
                    "contents": doc.get("contents", ""),
                    "score": round(score, 4)
                })

            return {"results": docs, "message": "Success"}

        except Exception as e:
            return {"results": [], "message": f"Error: {str(e)}"}

        finally:
            async with retriever_lock:
                available_retrievers.append(retriever_idx)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "retrievers": {
            "total": len(retriever_list),
            "available": len(available_retrievers)
        }
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml")
    parser.add_argument("--num_retriever", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    init_retrievers(args.config, args.num_retriever)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
