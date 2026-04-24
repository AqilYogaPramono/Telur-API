from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from APP.services.egg_analysis_service import enqueue_production_analysis_job
from APP.services.egg_analysis_service import get_production_analysis_job_state
from APP.utils.upload_validation import UploadValidationError, validate_image_upload

router = APIRouter(tags=["Egg Analysis - Production"])

class AnalyzeEggQueuedResponse(BaseModel):
    job_id: int


class AnalyzeEggJobStatusResponse(BaseModel):
    status: Literal["queued", "running", "succeeded", "failed"]
    result_id: int | None = None
    error: str | None = None


def _map_upload_validation_error(error: UploadValidationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.message)


@router.get("/analyze-egg/jobs/{job_id}", response_model=AnalyzeEggJobStatusResponse)
async def get_analyze_egg_job_status(job_id: int):
    try:
        state = await get_production_analysis_job_state(job_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error
    return AnalyzeEggJobStatusResponse(
        status=state.get("status", "queued"),
        result_id=state.get("result_id"),
        error=state.get("error"),
    )


@router.post("/analyze-egg")
async def analyze_egg_production(
    file: UploadFile = File(...),
):
    image_bytes = await file.read()
    try:
        validate_image_upload(file.filename or "", file.content_type, image_bytes)
    except UploadValidationError as error:
        raise _map_upload_validation_error(error) from error
    job_id = await enqueue_production_analysis_job(image_bytes)
    return AnalyzeEggQueuedResponse(job_id=job_id)

