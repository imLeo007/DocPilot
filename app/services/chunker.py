import re

from typing import Any

from chonkie import RecursiveChunker

from app.schemas.retriever import Chunk

from app.core.config import settings


chunker = RecursiveChunker(
    tokenizer=settings.embedding_model,
    chunk_size=400,
)


def get_heading(text: str) -> str | None:
    match = re.search(r"^(#{1,4})\s+(.+)$", text, flags=re.MULTILINE)
    
    if match:
        return match.group(2).strip()
    
    return None


def chunk_document(document: list[dict[str, Any]]) -> list[Chunk]:
    grouped_chunks: list[Chunk] = []

    index = 1

    for page_data in document:
        raw_text = page_data.get("text", "").strip()
        
        if not raw_text:
            continue

        metadata = page_data.get("metadata", {})
        
        page_no = metadata.get("page", 1)

        chunks = chunker(raw_text)

        heading = get_heading(raw_text)

        for chunk in chunks:
            chunk_text = chunk.text.strip()
            
            if not chunk_text:
                continue

            sub_heading = get_heading(chunk_text)

            if sub_heading:
                sub_heading = heading

            grouped_chunks.append(
                {
                    "text": chunk_text,
                    "headings": heading,
                    "page_no": page_no,
                    "chunk_index": index
                }
            )

            index += 1

    return grouped_chunks