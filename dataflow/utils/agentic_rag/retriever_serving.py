import os
import uvicorn
import argparse
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dataflow.serving.flash_rag_serving import FlashRAGServing

rag_service: Optional[FlashRAGServing] = None

class QueryRequest(BaseModel):
    query: str
    topk: int = 5

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    
    config_path = os.environ.get("RAG_CONFIG_PATH", "./retriever_config.yaml")
    
    max_workers_str = os.environ.get("RAG_MAX_WORKERS", "1")
    try:
        max_workers = int(max_workers_str)
    except ValueError:
        max_workers = 1

    print(f"[Lifespan] Initializing FlashRAG Service...")
    print(f"  - Config: {config_path}")
    print(f"  - Max Workers: {max_workers}")
    
    try:
        rag_service = await FlashRAGServing.create(
            config_path=config_path,
            max_workers=max_workers
        )
        
        rag_service.start_serving()
        print("[Lifespan] Service is ready.")
        
    except Exception as e:
        print(f"[Lifespan] Error initializing service: {e}")
        raise e

    yield

    print("[Lifespan] Shutting down...")
    if rag_service:
        await rag_service.cleanup()

app = FastAPI(lifespan=lifespan)

@app.post("/retrieve")
async def retrieve_docs(request_data: QueryRequest):
    global rag_service
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service is initializing.")

    query = request_data.query
    topk = request_data.topk

    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        results = await rag_service.retrieve_for_api(query, topk)
        if not results:
            return {"results": [], "message": "No results found."}
        return {"results": results, "message": "Success"}

    except Exception as e:
        print(f"Error processing request: {e}")
        return {"results": [], "message": f"Internal Error: {str(e)}"}

@app.get("/health")
async def health_check():
    is_ready = rag_service is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "config": os.environ.get("RAG_CONFIG_PATH", "unknown"),
        "max_workers": os.environ.get("RAG_MAX_WORKERS", "unknown")
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./retriever_config.yaml", help="Path to retriever config yaml")
    parser.add_argument("--max_workers", type=int, default=1, help="Number of threads for concurrent retrieval")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    os.environ["RAG_CONFIG_PATH"] = args.config
    os.environ["RAG_MAX_WORKERS"] = str(args.max_workers)
    
    print(f"Starting server on port {args.port} with {args.max_workers} workers...")
    uvicorn.run(app, host="0.0.0.0", port=args.port)