"""
Base configuration for database models.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from backend.src.enoro.core.config import settings

# Database setup
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.database_url
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Database dependency
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create tables
def create_tables():
    """Create all database tables."""
    # Import all models to register them with SQLAlchemy
    from . import video, channel, search, tags  # noqa: F401

    # Create all tables
    Base.metadata.create_all(bind=engine)
