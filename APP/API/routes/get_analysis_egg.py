from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from APP.core.database import get_db
from APP.models.egg_detections import EggDetection

router = APIRouter(tags=["Egg Analysis - Records"])


class EggDetectionSummary(BaseModel):
    id: int
    images_detection: str
    egg_count: int
    fertile_count: int
    infertile_count: int
    dead_count: int
    detected_at: datetime


@router.get("/egg-analysis", response_model=list[EggDetectionSummary])
async def get_egg_analysis(db: Session = Depends(get_db)):
    rows = db.scalars(
        select(EggDetection).order_by(
            EggDetection.detected_at.desc(),
            EggDetection.id.desc(),
        )
    ).all()
    return [
        EggDetectionSummary(
            id=row.id,
            images_detection=row.images_detection,
            egg_count=row.egg_count,
            fertile_count=row.fertile_count,
            infertile_count=row.infertile_count,
            dead_count=row.dead_count,
            detected_at=row.detected_at,
        )
        for row in rows
    ]

