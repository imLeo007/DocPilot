from typing import Any

from pydantic import BaseModel, Field

from typing import TypedDict


class Chunk(TypedDict):
    text: str
    headings: str
    page_no: int
    index: int


RetrievedChunks = dict[str, Any]


class RankedResponse(BaseModel):
    chunk_id: int
    document_id: int
    chunk_index: int
    page_no: int
    headings: str | None = None
    filename: str
    text: str
    vector_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None


class ToolRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: int | None = None
    category: str | None = None


class ToolResponse(BaseModel):
    query: str
    results: list[RankedResponse]