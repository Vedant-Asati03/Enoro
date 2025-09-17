"""
Main API router for Enoro.
"""

from fastapi import APIRouter

from .auth.youtube import router as youtube_auth_router
from .routers.ml import router as ml_router

# Create main API router
api_router = APIRouter()

# Include sub-routers
api_router.include_router(youtube_auth_router)
api_router.include_router(ml_router)


# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Enoro Video Discovery", "version": "1.0.0"}
