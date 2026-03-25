from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text

from backend.app.api import datasets, drive, inference, internal_storage, jobs, models
from backend.db.database import sync_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    from backend.db.orm import SyncOrm

    try:
        with sync_engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE models ADD COLUMN IF NOT EXISTS task_type TEXT")
            )
            conn.execute(
                text(
                    "ALTER TABLE models ADD COLUMN IF NOT EXISTS trained_at TIMESTAMPTZ"
                )
            )
    except Exception:
        pass

    SyncOrm.create_tables()
    try:
        from backend.app.services.storage import ensure_buckets
        ensure_buckets()
    except Exception:
        pass
    yield


app = FastAPI(
    title="SimpleAutoML API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(drive.router, prefix="/api/drive", tags=["drive"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(
    internal_storage.router,
    prefix="/api/internal/storage",
    tags=["internal"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}
