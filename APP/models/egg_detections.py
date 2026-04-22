from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from APP.core.database import Base

if TYPE_CHECKING:
    from APP.models.egg_classifications import EggClassification


class EggDetection(Base):
    __tablename__ = "egg_detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    images_detection: Mapped[str] = mapped_column(String(255), nullable=False)
    egg_count: Mapped[int] = mapped_column(Integer, nullable=False)
    fertile_count: Mapped[int] = mapped_column(Integer, nullable=False)
    infertile_count: Mapped[int] = mapped_column(Integer, nullable=False)
    dead_count: Mapped[int] = mapped_column(Integer, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.current_timestamp(),
    )

    classifications: Mapped[list["EggClassification"]] = relationship(
        "EggClassification",
        back_populates="detection",
        cascade="all, delete-orphan",
    )
