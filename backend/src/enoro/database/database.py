"""
Database configuration and initialization.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.src.enoro.core.config import settings

# Create database engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.database_url
    else {},
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()


def init_db():
    """Initialize database tables."""
    # Import models to register them with SQLAlchemy
    from .models.base import create_tables

    # Create all tables
    create_tables()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
