import re

from app.schemas.retriever import RetrievedChunks

from app.models.chunks import Chunks

from app.models.document import Document

from app.ai.reranker import rerank_chunks

from app.ai.embedding import create_question_embeddings

from sqlalchemy import func, select

from sqlalchemy.ext.asyncio import AsyncSession


async def retrieve_vector_chunks(
    db: AsyncSession,
    query_vector: list[float],
    candidate_k: int,
    document_id: int | None = None,
    category: str | None = None,
) -> list[RetrievedChunks]:
    distance = (
        1 - Chunks.embeddings.cosine_distance(query_vector)
    ).label("vector_score")

    query = (
        select(
            Chunks.id.label("chunk_id"),
            Chunks.document_id,
            Chunks.chunk_index,
            Chunks.page_no,
            Chunks.headings,
            Chunks.text,
            Document.filename,
            distance
        )
        .join(Document, Document.id == Chunks.document_id)
    )

    if document_id is not None:
        query = query.where(Chunks.document_id==document_id)

    if category is not None:
        query = query.where(Document.category == category)

    query = query.order_by(
        Chunks.embeddings.cosine_distance(query_vector)
    ).limit(candidate_k)

    result = await db.execute(query)

    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page_no": chunk.page_no,
            "headings": chunk.headings,
            "filename": chunk.filename,
            "text": chunk.text,
            "vector_score": round(float(chunk.vector_score), 4),
            "keyword_score": None,
        }
        for chunk in result.all()
    ]


async def retrieve_keyword_chunks(
    db: AsyncSession,
    question: str,
    candidate_k: int,
    document_id: int | None = None,
    category: str | None = None,
) -> list[RetrievedChunks]:

    tokens = re.findall(r"\w+", question.lower())

    if not tokens:
        return []

    tsquery = func.websearch_to_tsquery("english", " OR ".join(tokens))

    keyword_score = func.ts_rank_cd(Chunks.search_vector, tsquery).label("keyword_score")

    query = (
        select(
            Chunks.id.label("chunk_id"),
            Chunks.document_id,
            Chunks.chunk_index,
            Chunks.page_no,
            Chunks.headings,
            Chunks.text,
            Document.filename,
            keyword_score,
        )
        .join(Document, Document.id == Chunks.document_id)
        .where(Chunks.search_vector.op("@@")(tsquery))
    )

    if document_id is not None:
        query = query.where(Chunks.document_id == document_id)

    if category is not None:
        query = query.where(Document.category == category)

    query = query.order_by(keyword_score.desc()).limit(candidate_k)

    result = await db.execute(query)

    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "page_no": chunk.page_no,
            "headings": chunk.headings,
            "filename": chunk.filename,
            "text": chunk.text,
            "vector_score": None,
            "keyword_score": round(float(chunk.keyword_score), 4),
        }
        for chunk in result.all()
    ]


def fuse_rankings(
    vector_results: list[RetrievedChunks],
    keyword_results: list[RetrievedChunks],
    top_k: int,
    rrf_constant: int = 60,
) -> list[RetrievedChunks]:
    fused_results: dict[int, RetrievedChunks] = {}

    for rank, item in enumerate(vector_results, start=1):
        chunk_id = item["chunk_id"]

        fused_results[chunk_id] = item.copy()

        fused_results[chunk_id]["fusion_score"] = 1 /(rrf_constant + rank)

    for rank, item in enumerate(keyword_results, start=1):
        chunk_id = item["chunk_id"]

        rrf_score = (1/(rrf_constant + rank))

        if chunk_id in fused_results:

            fused_results[chunk_id]["keyword_score"] = item["keyword_score"]

            fused_results[chunk_id]["fusion_score"] += rrf_score

        else:

            fused_results[chunk_id] = item.copy()

            fused_results[chunk_id]["fusion_score"] = rrf_score

    ranked = sorted(
        fused_results.values(),
        key=lambda item: float(item["fusion_score"] or 0),
        reverse=True
    )

    return ranked[:top_k]


async def retrieve_ranked_chunks(
    db: AsyncSession,
    query: str,
    top_k: int = 5,
    document_id: int | None = None,
    category: str | None = None,
    candidate_k: int = 25,
) -> list[RetrievedChunks]:
    query_vector = create_question_embeddings(query)

    vector_chunks = await retrieve_vector_chunks(
        db=db,
        query_vector=query_vector,
        candidate_k=candidate_k,
        document_id=document_id,
        category=category,
    )

    keyword_chunks = await retrieve_keyword_chunks(
        db=db,
        question=query,
        candidate_k=candidate_k,
        document_id=document_id,
        category=category,
    )

    fused_chunks = fuse_rankings(
        vector_results=vector_chunks,
        keyword_results=keyword_chunks,
        top_k=candidate_k,
    )

    if not fused_chunks:
        return []

    return rerank_chunks(
        question=query,
        chunks=fused_chunks,
        top_k=top_k
    )