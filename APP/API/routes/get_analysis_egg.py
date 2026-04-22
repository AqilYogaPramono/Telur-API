from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from APP.core.database import get_db
from APP.models.egg_classifications import EggClassification
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


class EggClassificationItem(BaseModel):
    egg_index: int
    classification_label: str
    confidence_score: float


class EggDetectionDetail(EggDetectionSummary):
    egg_classifications: list[EggClassificationItem]


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


@router.get("/egg-analysis/{id}", response_model=EggDetectionDetail)
async def get_egg_analysis_by_id(id: int, db: Session = Depends(get_db)):
    row = db.scalar(
        select(EggDetection)
        .options(selectinload(EggDetection.classifications))
        .where(EggDetection.id == id)
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Egg analysis not found.")
    classifications = sorted(row.classifications, key=lambda x: x.egg_index)
    return EggDetectionDetail(
        id=row.id,
        images_detection=row.images_detection,
        egg_count=row.egg_count,
        fertile_count=row.fertile_count,
        infertile_count=row.infertile_count,
        dead_count=row.dead_count,
        detected_at=row.detected_at,
        egg_classifications=[
            EggClassificationItem(
                egg_index=item.egg_index,
                classification_label=item.classification_label,
                confidence_score=float(item.confidence_score),
            )
            for item in classifications
        ],
    )

