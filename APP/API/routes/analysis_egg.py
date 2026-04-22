import asyncio
from decimal import Decimal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from APP.core.database import get_db
from APP.models.egg_classifications import EggClassification
from APP.models.egg_detections import EggDetection
from APP.services.gdrive_service import upload_production_analysis
from APP.services.inference_runner import run_sync_with_inference_timeout
from APP.services.predict_egg import analyze_egg_yolo_crop_sync
from APP.utils.upload_validation import UploadValidationError, validate_image_upload

router = APIRouter(tags=["Egg Analysis - Production"])


def _map_upload_validation_error(error: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)


def _map_inference_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, TimeoutError):
        return HTTPException(
            status_code=504,
            detail="Inference timeout. The prediction process exceeded 5 minutes.",
        )
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=f"Prediction failed: {error}")


@router.post("/analyze-egg")
async def analyze_egg_production(db: Session = Depends(get_db), file: UploadFile = File(...)):
    image_bytes = await file.read()
    try:
        validate_image_upload(file.filename or "", file.content_type, image_bytes)
    except UploadValidationError as error:
        raise _map_upload_validation_error(error) from error

    try:
        analysis = await run_sync_with_inference_timeout(lambda: analyze_egg_yolo_crop_sync(image_bytes))
    except TimeoutError as error:
        raise _map_inference_http_exception(error) from error
    except ValueError as error:
        raise _map_inference_http_exception(error) from error
    except Exception as error:
        raise _map_inference_http_exception(error) from error

    crop_items = [
        {
            "egg_index": item.egg_index,
            "label": item.classification.classification_label,
            "png_bytes": item.png_bytes,
        }
        for item in analysis.crop_previews
    ]
    try:
        raw = await run_in_threadpool(upload_production_analysis, analysis.overlay_jpeg_bytes, crop_items)
    except Exception as error:
        raise HTTPException(status_code=502, detail=f"Google Drive upload failed: {error}") from error

    if not raw.get("uploaded_any"):
        raise HTTPException(status_code=502, detail="Google Drive upload failed: no files uploaded.")

    file_ids: dict[str, str] = raw["drive_file_ids"]
    primary_image_id = (file_ids.get("bounding_box") or next(iter(file_ids.values()), ""))[:255]

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
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database persist failed: {error}") from error

    return {"id": row.id, "message": "Analysis saved successfully."}

