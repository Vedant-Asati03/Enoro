"""
FastAPI application entry point for Enoro Video Discovery.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from backend.src.enoro.core.config import settings
from backend.src.enoro.database.database import init_db
from backend.src.enoro.api.router import api_router
from backend.src.enoro.api.middleware.auth import AuthenticationMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("🚀 Starting Enoro Video Discovery API...")

    # Initialize database
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        raise

    yield

    # Shutdown
    print("🛑 Shutting down Enoro Video Discovery API...")


# Create FastAPI application
app = FastAPI(
    title="Enoro Video Discovery API",
    description="AI-powered video recommendation system with enhanced features",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
app.add_middleware(AuthenticationMiddleware)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
            "message": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


# Include routers
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enoro Video Discovery API",
        "version": "1.0.0",
        "description": "AI-powered video recommendation system",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
    )
