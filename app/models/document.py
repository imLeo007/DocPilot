from datetime import datetime

from typing import TYPE_CHECKING

from sqlalchemy import String, Integer, DateTime, func

from sqlalchemy.orm import relationship, mapped_column, Mapped

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chunks import Chunks


class Document(Base):
    __tablename__ = "document"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chunks: Mapped[list["Chunks"]] = relationship(
        "Chunks",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )