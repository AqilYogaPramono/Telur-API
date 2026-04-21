from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from APP.services.experiment_direct_analysis import run_direct_cnn_classification
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


def _map_upload_validation_error(error: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)


async def _read_upload_bytes(file: UploadFile) -> bytes:
    return await file.read()


@router.post("/analyze-egg/direct", response_model=AnalyzeEggDirectResponse)
async def analyze_egg_experiment_direct(file: UploadFile = File(...)):
    clear_public_preview_files(PUBLIC_DIR)
    image_bytes = await _read_upload_bytes(file)
    try:
        validate_image_upload(file.filename or "", file.content_type, image_bytes)
    except UploadValidationError as error:
        raise _map_upload_validation_error(error) from error
    try:
        result = await run_direct_cnn_classification(image_bytes)
    except TimeoutError as error:
        raise HTTPException(
            status_code=504,
            detail="Inference timeout. The prediction process exceeded 5 minutes.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}") from error
    return AnalyzeEggDirectResponse(
        egg_classifications=EggClassificationDirectBody(
            classification_label=result.classification_label,
            confidence_score=result.confidence_score,
        )
    )
