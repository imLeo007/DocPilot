from app.core.database import get_db

from fastapi import Depends, APIRouter

from typing import Annotated

from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.retriever import retrieve_ranked_chunks

from app.schemas.retriever import ToolRequest, ToolResponse, RankedResponse



router = APIRouter(prefix="/question", tags=["Question", "Search"])


DBSession = Annotated[AsyncSession, Depends(get_db)]


@router.post("/search", response_model=ToolResponse)
async def search_content(request: ToolRequest, db: DBSession) -> ToolResponse:
    chunks = await retrieve_ranked_chunks(
        db=db,
        query=request.query,
        document_id=request.document_id,
        category=request.category,
        top_k=request.top_k
    )

    results = [
        RankedResponse.model_validate(chunk)
        for chunk in chunks
    ]

    return ToolResponse(
        query=request.query,
        results=results
    )