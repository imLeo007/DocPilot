from datetime import datetime

from typing import Any, TYPE_CHECKING

from sqlalchemy import (
    Text,
    Integer,
    func,
    DateTime,
    Computed,
    ForeignKey,
    Index,
    UniqueConstraint,
)

from sqlalchemy.orm import mapped_column, Mapped, relationship

from sqlalchemy.dialects.postgresql import TSVECTOR

from pgvector.sqlalchemy import Vector

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.document import Document


class Chunks(Base):
    __tablename__ = "chunks"

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk_idx"),
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
        Index(
            "ix_chunks_embeddings_hnsw",
            "embeddings",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embeddings": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    document_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    page_no: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    headings: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embeddings: Mapped[Any] = mapped_column(
        Vector(384),
        nullable=False,
    )

    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('english', coalesce(headings, '') || ' ' || coalesce(text, ''))",
            persisted=True,
        ),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )