from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from APP.services.egg_classification_experiment import classify_egg_from_image_bytes
from APP.services.experiment_yolo_crop_analysis import run_yolo_crop_experiment_sync
from APP.services.inference_runner import run_sync_with_inference_timeout
from APP.utils.public_preview import clear_public_preview_files
from APP.utils.upload_validation import UploadValidationError, validate_image_upload

APP_DIR = Path(__file__).resolve().parents[2]
PUBLIC_DIR = APP_DIR / "public"

router = APIRouter(prefix="/experiment", tags=["Egg Analysis - Experiment"])


class EggClassificationDirectBody(BaseModel):
    classification_label: str
    confidence_score: float


class AnalyzeEggDirectResponse(BaseModel):
    egg_classifications: EggClassificationDirectBody


class EggClassificationYoloItem(BaseModel):
    egg_index: int
    classification_label: str
    confidence_score: float


class EggDetectionsYoloCropBody(BaseModel):
    images_detection: str
    egg_count: int
    fertile_count: int
    infertile_count: int
    dead_count: int
    egg_classifications: list[EggClassificationYoloItem]


class AnalyzeEggYoloCropResponse(BaseModel):
    egg_detections: EggDetectionsYoloCropBody


def _map_upload_validation_error(error: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)


async def _read_upload_bytes(file: UploadFile) -> bytes:
    return await file.read()


def _map_inference_http_exception(error: Exception) -> HTTPException:
    if isinstance(error, TimeoutError):
        return HTTPException(
            status_code=504,
            detail="Inference timeout. The prediction process exceeded 5 minutes.",
        )
    if isinstance(error, ValueError):
        return HTTPException(status_code=400, detail=str(error))
    return HTTPException(status_code=500, detail=f"Prediction failed: {error}")


@router.post("/analyze-egg/direct", response_model=AnalyzeEggDirectResponse)
async def analyze_egg_experiment_direct(file: UploadFile = File(...)):
    clear_public_preview_files(PUBLIC_DIR)
    image_bytes = await _read_upload_bytes(file)
    try:
        validate_image_upload(file.filename or "", file.content_type, image_bytes)
    except UploadValidationError as error:
        raise _map_upload_validation_error(error) from error
    try:
        result = await run_sync_with_inference_timeout(
            lambda: classify_egg_from_image_bytes(image_bytes),
        )
    except TimeoutError as error:
        raise _map_inference_http_exception(error) from error
    except ValueError as error:
        raise _map_inference_http_exception(error) from error
    except Exception as error:
        raise _map_inference_http_exception(error) from error
    return AnalyzeEggDirectResponse(
        egg_classifications=EggClassificationDirectBody(
            classification_label=result.classification_label,
            confidence_score=result.confidence_score,
        )
    )


@router.post("/analyze-egg/yolo-crop", response_model=AnalyzeEggYoloCropResponse)
async def analyze_egg_experiment_yolo_crop(file: UploadFile = File(...)):
    clear_public_preview_files(PUBLIC_DIR)
    image_bytes = await _read_upload_bytes(file)
    try:
        validate_image_upload(file.filename or "", file.content_type, image_bytes)
    except UploadValidationError as error:
        raise _map_upload_validation_error(error) from error
    upload_name = file.filename or "upload"
    try:
        outcome = await run_sync_with_inference_timeout(
            lambda: run_yolo_crop_experiment_sync(image_bytes, upload_name, PUBLIC_DIR),
        )
    except TimeoutError as error:
        raise _map_inference_http_exception(error) from error
    except ValueError as error:
        raise _map_inference_http_exception(error) from error
    except Exception as error:
        raise _map_inference_http_exception(error) from error
    return AnalyzeEggYoloCropResponse(
        egg_detections=EggDetectionsYoloCropBody(
            images_detection=outcome.images_detection,
            egg_count=outcome.egg_count,
            fertile_count=outcome.fertile_count,
            infertile_count=outcome.infertile_count,
            dead_count=outcome.dead_count,
            egg_classifications=[
                EggClassificationYoloItem(
                    egg_index=row.egg_index,
                    classification_label=row.classification_label,
                    confidence_score=row.confidence_score,
                )
                for row in outcome.egg_classifications
            ],
        )
    )
