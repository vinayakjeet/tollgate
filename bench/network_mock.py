"""A local OpenAI-shaped HTTP server with a fixed delay: the network_mock
equivalent LiteLLM's protocol uses (BACKLOG M5.1).

Two mocks exist because they answer different questions. The in-process mock has
no network and no jitter, so its numbers isolate Tollgate's own code. This one
puts a real socket, a real round trip and real serialisation back in the path;
the gap between the two rows is the part of "gateway overhead" that is actually
transport. Both are deterministic: fixed delay, no randomness, so variance
measured against them belongs to Tollgate or to nothing.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI, Response


class NetworkMock:
    """Runs uvicorn in a daemon thread. `wait_ready` blocks until the socket
    answers, because a benchmark that races its own fixture measures nothing."""

    def __init__(self, delay_s: float = 0.02, chunk_delay_s: float = 0.004) -> None:
        self.delay_s = delay_s
        self.chunk_delay_s = chunk_delay_s
        self.port = _free_port()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        app = FastAPI()

        @app.get("/healthz")
        async def healthz():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        async def completions(payload: dict):
            await asyncio.sleep(self.delay_s)
            if payload.get("stream"):
                from fastapi.responses import StreamingResponse

                return StreamingResponse(
                    self._chunk_frames(), media_type="text/event-stream"
                )
            prompt = payload["messages"][-1]["content"]
            return Response(
                content=_json_body(f"network mock echo: {prompt}", len(prompt.split())),
                media_type="application/json",
            )

        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error")
        )

        def _serve():
            # Windows defaults to the Proactor loop, which has been observed to
            # fail transport allocation under rapid connect/disconnect churn.
            # The selector loop is boring, and boring is what a fixture owes.
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            self._server.run()

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()
        self.wait_ready()

    def wait_ready(self, timeout_s: float = 10.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                import httpx as _httpx

                _httpx.get(f"http://127.0.0.1:{self.port}/healthz", timeout=0.25)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("network_mock never became ready")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)

    # The provider layer posts {base_url}/chat/completions, hence /v1 here.
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _chunk_frames(self):
        async def gen():
            import json

            for word in ["the", "quick", "brown", "fox", "jumps"]:
                await asyncio.sleep(self.chunk_delay_s)
                frame = {
                    "id": "chatcmpl-nm",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "network-mock",
                    "choices": [
                        {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(frame)}\n\n"
            yield "data: [DONE]\n\n"

        return gen()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _json_body(text: str, tokens_in: int) -> bytes:
    import json

    payload = {
        "id": "chatcmpl-nm",
        "object": "chat.completion",
        "created": 0,
        "model": "network-mock",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_in + 2},
    }
    return json.dumps(payload).encode()
