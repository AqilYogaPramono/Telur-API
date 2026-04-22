from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR.parents[1]
load_dotenv(_REPO_ROOT / ".env", encoding="utf-8-sig", override=True)
load_dotenv(_APP_DIR / ".env", encoding="utf-8-sig", override=True)

from APP.API.routes.analysis_egg_experiment import router as egg_experiment_router
from APP.API.routes.analysis_egg import router as egg_production_router
from APP.API.routes.database_management import router as database_management_router
from APP.API.routes.get_analysis_egg import router as egg_records_router

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Telur API", version="1.0.0")
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")
app.include_router(egg_experiment_router)
app.include_router(egg_production_router)
app.include_router(database_management_router)
app.include_router(egg_records_router)


@app.get("/", include_in_schema=False)
async def main():
    return {"message": "API is running."}
