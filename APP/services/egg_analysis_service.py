import asyncio
from decimal import Decimal
from typing import Any, Literal, TypedDict

from starlette.concurrency import run_in_threadpool

from APP.core.database import get_sessionmaker
from APP.models.egg_classifications import EggClassification
from APP.models.egg_detections import EggDetection
from APP.services.gdrive_service import upload_production_analysis
from APP.services.inference_runner import run_sync_with_inference_timeout
from APP.services.predict_egg import analyze_egg_yolo_crop_sync


class JobState(TypedDict, total=False):
    status: Literal["queued", "running", "succeeded", "failed"]
    result_id: int
    error: str


_job_lock = asyncio.Lock()
_next_job_id = 1
_jobs: dict[int, JobState] = {}


async def enqueue_production_analysis_job(image_bytes: bytes) -> int:
    global _next_job_id
    async with _job_lock:
        job_id = _next_job_id
        _next_job_id += 1
        _jobs[job_id] = {"status": "queued"}
    asyncio.create_task(_run_production_analysis_job(job_id, image_bytes))
    return job_id


async def get_production_analysis_job_state(job_id: int) -> JobState:
    async with _job_lock:
        state = _jobs.get(job_id)
        if state is None:
            raise KeyError(f"Job {job_id} not found.")
        return dict(state)


async def _run_production_analysis_job(job_id: int, image_bytes: bytes) -> None:
    async with _job_lock:
        _jobs[job_id] = {"status": "running"}
    try:
        analysis = await run_sync_with_inference_timeout(lambda: analyze_egg_yolo_crop_sync(image_bytes))
        crop_items: list[dict[str, Any]] = [
            {
                "egg_index": item.egg_index,
                "label": item.classification.classification_label,
                "png_bytes": item.png_bytes,
            }
            for item in analysis.crop_previews
        ]
        raw = await run_in_threadpool(upload_production_analysis, analysis.overlay_jpeg_bytes, crop_items)
        if not raw.get("uploaded_any"):
            raise RuntimeError("Google Drive upload failed: no files uploaded.")
        file_ids: dict[str, str] = raw["drive_file_ids"]
        primary_image_id = (file_ids.get("bounding_box") or next(iter(file_ids.values()), ""))[:255]
        SessionLocal = get_sessionmaker()
        db = SessionLocal()
        try:
            row = EggDetection(
                images_detection=primary_image_id,
                egg_count=analysis.egg_count,
                fertile_count=analysis.fertile_count,
                infertile_count=analysis.infertile_count,
                dead_count=analysis.dead_count,
            )
            db.add(row)
            db.flush()
            for item in analysis.crop_previews:
                db.add(
                    EggClassification(
                        detection_id=row.id,
                        egg_index=item.egg_index,
                        classification_label=item.classification.classification_label,
                        confidence_score=Decimal(str(round(float(item.classification.confidence_score), 4))),
                    )
                )
            db.commit()
            db.refresh(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
        async with _job_lock:
            _jobs[job_id] = {"status": "succeeded", "result_id": row.id}
    except Exception as error:
        async with _job_lock:
            _jobs[job_id] = {"status": "failed", "error": str(error)}
