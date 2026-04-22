from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
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


def to_egg_detection_summary(row: EggDetection) -> EggDetectionSummary:
    return EggDetectionSummary(
        id=row.id,
        images_detection=row.images_detection,
        egg_count=row.egg_count,
        fertile_count=row.fertile_count,
        infertile_count=row.infertile_count,
        dead_count=row.dead_count,
        detected_at=row.detected_at,
    )


@router.get("/egg-analysis", response_model=list[EggDetectionSummary])
async def get_egg_analysis(db: Session = Depends(get_db)):
    rows = db.scalars(
        select(EggDetection).order_by(
            EggDetection.detected_at.desc(),
            EggDetection.id.desc(),
        )
    ).all()
    return [to_egg_detection_summary(row) for row in rows]


@router.get("/egg-analysis-news", response_model=EggDetectionSummary)
async def get_egg_analysis_news(db: Session = Depends(get_db)):
    today = date.today()
    row = db.scalar(
        select(EggDetection)
        .where(func.date(EggDetection.detected_at) == today)
        .order_by(
            EggDetection.detected_at.desc(),
            EggDetection.id.desc(),
        )
        .limit(1)
    )
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Egg analysis news for today not found.",
        )
    return to_egg_detection_summary(row)


@router.get("/egg-analysis-news/history", response_model=list[EggDetectionSummary])
async def get_egg_analysis_news_history(db: Session = Depends(get_db)):
    today = date.today()
    start_date = today - timedelta(days=4)
    end_date = today - timedelta(days=1)

    rows = db.scalars(
        select(EggDetection)
        .where(func.date(EggDetection.detected_at).between(start_date, end_date))
        .order_by(
            EggDetection.detected_at.desc(),
            EggDetection.id.desc(),
        )
    ).all()

    latest_by_day: dict[date, EggDetectionSummary] = {}
    for row in rows:
        detection_day = row.detected_at.date()
        if detection_day not in latest_by_day:
            latest_by_day[detection_day] = to_egg_detection_summary(row)

    return list(latest_by_day.values())


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

