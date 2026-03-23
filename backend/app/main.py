from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import datasets, drive, jobs, models


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB tables, MinIO buckets
    from backend.db.orm import SyncOrm

    SyncOrm.create_tables()
    try:
        from backend.app.services.storage import ensure_buckets
        ensure_buckets()
    except Exception:
        pass  # MinIO may not be ready yet
    yield
    # Shutdown
    pass


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
app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
