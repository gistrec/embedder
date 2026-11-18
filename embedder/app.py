"""HTTP API эмбеддингов.

    POST /v1/embed  {"input": ["текст", ...], "type": "query" | "document"}
                 -> {"model": "...", "dim": 384, "data": [[...], ...]}
    GET  /health    -> {"status": "ok", "model": "...", "dim": 384}

Модель грузится один раз при старте; вычисление — в пуле потоков, по одному запросу
за раз (ONNX Runtime и так использует ``EMBEDDER_THREADS`` ядер).
"""
import asyncio
import hmac
import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from embedder import model as model_mod
from embedder.config import settings

logger = logging.getLogger("embedder")


class EmbedRequest(BaseModel):
    input: list[str] = Field(min_length=1)
    type: Literal["query", "document"] = "document"


class EmbedResponse(BaseModel):
    model: str
    dim: int
    data: list[list[float]]


def create_app(loader=model_mod.load) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        started = time.monotonic()
        app.state.embedder = await run_in_threadpool(loader)
        app.state.lock = asyncio.Lock()
        if not settings.token:
            logger.warning("EMBEDDER_TOKEN пуст: /v1/embed отвечает без авторизации")
        logger.warning(
            "model %s loaded: dim=%s in %.1fs",
            app.state.embedder.name, app.state.embedder.dim, time.monotonic() - started,
        )
        yield

    app = FastAPI(title="embedder", lifespan=lifespan)

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.token:
            return
        # Сравниваем байты: compare_digest на str с не-ASCII бросает TypeError (было бы 500).
        expected = f"Bearer {settings.token}".encode()
        if not authorization or not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(status_code=401, detail="invalid token")

    @app.get("/health")
    async def health(request: Request):
        e = request.app.state.embedder
        return {"status": "ok", "model": e.name, "dim": e.dim}

    @app.post("/v1/embed", response_model=EmbedResponse, dependencies=[Depends(auth)])
    async def embed(body: EmbedRequest, request: Request):
        if len(body.input) > settings.max_inputs:
            raise HTTPException(status_code=413, detail=f"at most {settings.max_inputs} inputs per request")
        if any(len(t) > settings.max_chars for t in body.input):
            raise HTTPException(status_code=413, detail=f"input longer than {settings.max_chars} chars")
        if any(not t.strip() for t in body.input):
            raise HTTPException(status_code=422, detail="empty input")
        e = request.app.state.embedder
        async with request.app.state.lock:
            vectors = await run_in_threadpool(e.embed, body.input, body.type)
        return EmbedResponse(model=e.name, dim=e.dim, data=vectors)

    return app


app = create_app()
