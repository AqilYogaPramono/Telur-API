import os

from fastapi import APIRouter, Header, HTTPException

from APP.core.database import Base, get_engine
from APP.models.egg_classifications import EggClassification
from APP.models.egg_detections import EggDetection 

router = APIRouter(tags=["Database Management"])


def _required_reset_token() -> str | None:
    value = (os.getenv("RESET_SCHEMA_TOKEN") or "").strip()
    return value or None


@router.post("/reset-schema")
async def reset_schema(ResetToken: str | None = Header(default=None, alias="Token")):
    required = _required_reset_token()
    if required is not None and ResetToken != required:
        raise HTTPException(status_code=401, detail="Invalid reset schema token.")
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    return {"message": "Schema has been reset."}

