"""
CareerPilot — FastAPI Application Entry Point (Phase 1)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure the project root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import get_settings

# ── Settings ──────────────────────────────────────────────────────────────────
settings = get_settings()

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=settings.log_level,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>"
    ),
)


from database.connection import init_db

# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # ── Startup ──────────────────────────────────────────────────────────────
    logger.info("CareerPilot API starting up…")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"LLM model   : {settings.llm_model}")
    logger.info(f"Docs        : http://{settings.app_host}:{settings.app_port}/docs")
    try:
        init_db()
        logger.info("SQLite persistent database ready.")
    except Exception as exc:
        logger.error(f"Failed to initialize SQLite database: {exc}")
    yield
    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("CareerPilot API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CareerPilot",
    description=(
        "An Agentic AI for Personalized Placement Preparation. "
        "Powered by LangGraph + Gemini."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from backend.routers.auth import router as auth_router  # noqa: E402
from backend.routers.agent import router as agent_router  # noqa: E402
from backend.routers.documents import router as documents_router  # noqa: E402
from backend.routers.students import router as students_router  # noqa: E402
from backend.routers.placement import router as placement_router  # noqa: E402
from backend.routers.interview import interview_router  # noqa: E402
app.include_router(auth_router)
app.include_router(agent_router)
app.include_router(documents_router)
app.include_router(students_router)
app.include_router(placement_router)
app.include_router(interview_router, prefix="/api/interview", tags=["Mock Interview Agent"])





# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Basic liveness probe — confirms the API is running."""
    return {
        "status": "ok",
        "app": "CareerPilot",
        "version": "0.2.0",
        "environment": settings.app_env,
    }


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root() -> dict:
    return {
        "message": "Welcome to CareerPilot API",
        "docs": "/docs",
        "health": "/health",
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )
