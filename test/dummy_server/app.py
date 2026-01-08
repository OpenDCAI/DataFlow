import json
import time
from flask import Flask, request, jsonify, Response, stream_with_context

def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"ok": True}

    def _parse_bool(v: str | None) -> bool:
        if v is None:
            return False
        return v.lower() in ("1", "true", "yes", "y", "on")

    @app.post("/v1/chat/completions")
    def chat_completions():
        """
        DeepSeek 行为模拟：
        - 高流量排队时：HTTP连接保持，持续发送 keep-alive
          非流式：持续发送空行 "\n"
          流式：持续发送 SSE 注释 ": keep-alive\n\n"
        - 如果 10 分钟(默认)后仍未开始推理：服务器关闭连接（这里用 generator 结束来模拟）
        - 开始推理后：返回最终 JSON（非流式）或 SSE data（流式）

        query 参数：
          - ?queue=秒数           模拟“开始推理前排队时长”（期间持续 keep-alive）
          - ?max_wait=秒数        最长等待推理开始（默认 600=10min），超出则断开
          - ?ka_interval=秒数     keep-alive 间隔（默认 1s）
          - ?status=500           直接返回错误码（不进入 keep-alive）
          - ?body=xxx             最终内容
          - ?think=xxx            最终 reasoning_content（可选）
          - ?stream=1             模拟流式（也会读取 payload 里的 stream 字段）
        """
        queue_s = float(request.args.get("queue", "0"))
        max_wait_s = float(request.args.get("max_wait", "600"))
        ka_interval = float(request.args.get("ka_interval", "1"))
        status = int(request.args.get("status", "200"))
        body = request.args.get("body", "dummy response")
        think = request.args.get("think", "dummy reasoning")

        payload = request.get_json(silent=True) or {}
        # stream 优先级：query 参数 > payload.stream
        stream_q = request.args.get("stream")
        stream_mode = _parse_bool(stream_q) if stream_q is not None else bool(payload.get("stream", False))

        # 如果强制错误，直接返回（模拟服务端立刻拒绝）
        if status != 200:
            return jsonify({"error": f"dummy error with status={status}"}), status

        def gen():
            start = time.monotonic()

            # 1) 排队阶段：持续 keep-alive，直到 queue_s 结束或 max_wait_s 到期
            while True:
                elapsed = time.monotonic() - start

                # 超过最长等待：断开连接（DeepSeek：10分钟仍未开始推理就关闭）
                if elapsed >= max_wait_s:
                    # 直接结束 generator => 服务器关闭连接
                    return

                # 到了 queue_s，进入“推理/返回”阶段
                if elapsed >= queue_s:
                    break

                # 发送 keep-alive
                if stream_mode:
                    # SSE keep-alive 注释
                    yield ": keep-alive\n\n"
                else:
                    # 非流式 keep-alive 空行
                    yield "\n"

                time.sleep(ka_interval)

            # 2) 推理完成：输出最终响应
            if not stream_mode:
                # 非流式：最终返回一个 JSON（但注意：前面已经输出过 body 的空行了，
                # 这在真实服务端实现里通常是“先发 header + chunked body”，最终也会是 JSON 文本。
                # 所以这里我们也用 chunked 输出 JSON 字符串。)
                final = {
                    "id": "dummy-chatcmpl-001",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": body,
                            "reasoning_content": think
                        }
                    }]
                }
                yield json.dumps(final, ensure_ascii=False)
            else:
                # 流式：用 SSE data 模拟 token 流 + 最后 [DONE]
                # 这里只做最简两段：先发一个 chunk，再发 DONE
                chunk = {
                    "id": "dummy-chatcmpl-001",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": body},
                    }]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

        # content-type：流式用 text/event-stream；非流式用 application/json（但仍然 chunked）
        mimetype = "text/event-stream" if stream_mode else "application/json"
        return Response(stream_with_context(gen()), mimetype=mimetype)

    @app.post("/v1/embeddings")
    def embeddings():
        """
        embeddings 一般不需要 keep-alive，但你也可以用同样逻辑模拟。
        这里保留你原逻辑。
        """
        delay = float(request.args.get("delay", "0"))
        status = int(request.args.get("status", "200"))
        dim = int(request.args.get("dim", "8"))

        if delay > 0:
            time.sleep(delay)

        if status != 200:
            return jsonify({"error": f"dummy error with status={status}"}), status

        return jsonify({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.01] * dim
            }]
        })

    return app
