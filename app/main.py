from time import perf_counter

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routers.question import router as question_router

from app.routers.document import router as document_router

from app.ai.reranker import load_reranker

from app.ai.embedding import create_question_embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):

    start = perf_counter()

    load_reranker()

    print("\nReranker is loaded:", perf_counter() - start, "\n")

    question = "example question"

    start = perf_counter()

    create_question_embeddings(question)

    print("\nEmbedding model is loaded: ", perf_counter() - start, "\n")

    yield


app = FastAPI(title="DocPilot", version="5.0.0", lifespan=lifespan)


app.include_router(document_router)

app.include_router(question_router)


@app.get("/")
async def root() -> dict:
    return {
        "message": "DocPilot is running."
    }