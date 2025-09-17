"""
Health check endpoints.
"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
import psutil
import sys

from ...core.config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    system_info: dict


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "memory_percent": memory.percent,
            "disk_total_gb": round(disk.total / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_percent": round((disk.used / disk.total) * 100, 2),
            "python_version": sys.version,
        }

        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            uptime_seconds=psutil.boot_time(),
            system_info=system_info,
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version=settings.app_version,
            uptime_seconds=0.0,
            system_info={"error": str(e)},
        )


@router.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes/Docker."""
    return {"status": "ready", "timestamp": datetime.utcnow()}


@router.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes/Docker."""
    return {"status": "alive", "timestamp": datetime.utcnow()}
