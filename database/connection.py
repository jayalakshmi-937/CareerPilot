"""
CareerPilot — SQLite Database Connection & Session Management

Configures SQLAlchemy engine, session maker, and database initialization
supporting persistent student preparation profiles and historical evaluation tracking.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import get_settings

settings = get_settings()

# Ensure SQLite directory exists
db_url = settings.database_url
if db_url.startswith("sqlite:///"):
    sqlite_path_str = db_url.replace("sqlite:///", "")
    sqlite_path = Path(sqlite_path_str)
    if not sqlite_path.is_absolute():
        sqlite_path = Path.cwd() / sqlite_path
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured SQLite database directory exists at: {sqlite_path.parent}")

engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database session with automatic cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for non-FastAPI standalone operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the database if they do not exist."""
    import database.models  # noqa: F401 - ensure models are registered
    Base.metadata.create_all(bind=engine)
    logger.info("SQLite database tables initialized successfully.")
