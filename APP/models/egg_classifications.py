from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from APP.core.database import Base

if TYPE_CHECKING:
    from APP.models.egg_detections import EggDetection


class EggClassification(Base):
    __tablename__ = "egg_classifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detection_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("egg_detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    egg_index: Mapped[int] = mapped_column(Integer, nullable=False)
    classification_label: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence_score: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    detection: Mapped["EggDetection"] = relationship(
        "EggDetection",
        back_populates="classifications",
    )
