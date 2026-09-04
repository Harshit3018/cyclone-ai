"""
CYCLONE-AI Backend Application
FastAPI-based backend for the Tropical Cyclone Intelligence Platform.
"""
import os
import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .core.config import settings
from .database.session import engine, create_tables
from .api import router as api_router

logger = logging.getLogger("cyclone_ai")

# Resolve frontend dist directory
_frontend_dir = PROJECT_ROOT / settings.FRONTEND_DIR
_serve_frontend = settings.SERVE_FRONTEND or os.getenv("SERVE_FRONTEND", "").lower() in ("true", "1", "yes")
if not _serve_frontend:
    # Also auto-detect: if frontend/dist exists and no dev proxy, serve it
    _serve_frontend = _frontend_dir.is_dir() and (_frontend_dir / "index.html").exists()

# Request paths with these suffixes are static assets. They must never fall through to
# the SPA shell — see serve_spa() below.
_ASSET_SUFFIXES = {
    ".js", ".mjs", ".css", ".map", ".json", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".webp", ".avif", ".woff", ".woff2", ".ttf", ".otf", ".eot", ".wasm",
    ".txt", ".xml", ".webmanifest",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("=" * 60)
    logger.info("CYCLONE-AI Backend Starting")
    logger.info(f"  Environment: {settings.APP_ENV}")
    logger.info(f"  Database: {settings.DATABASE_URL}")
    logger.info(f"  Serve Frontend: {_serve_frontend}")
    if _serve_frontend:
        logger.info(f"  Frontend Dir: {_frontend_dir}")
    logger.info("=" * 60)

    # Create database tables
    create_tables()
    logger.info("Database tables created/verified")

    # Seed demo data if needed
    from .services.seed_service import seed_demo_data
    seed_demo_data()

    # Initialize ML predictor (lazy loading)
    logger.info("ML predictor will be loaded on first inference request")

    yield

    logger.info("CYCLONE-AI Backend Shutting Down")


app = FastAPI(
    title="CYCLONE-AI API",
    description=(
        "Multi-Source AI Platform for Tropical Cyclone Detection, Classification, "
        "Intensity and Track Prediction. This is a research and decision-support system. "
        "It is NOT an official warning system."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# --- Static file serving for production ---
if _serve_frontend and _frontend_dir.is_dir():
    # Serve static assets (JS, CSS, images) from frontend/dist/assets
    _assets_dir = _frontend_dir / "assets"
    if _assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")

    # Serve other static files from frontend/dist (favicon, etc.)
    # But NOT as a catch-all — we need the SPA fallback below

    @app.get("/vite.svg")
    async def vite_svg():
        """Serve the Vite favicon."""
        svg_path = _frontend_dir / "vite.svg"
        if svg_path.exists():
            return FileResponse(str(svg_path), media_type="image/svg+xml")
        return JSONResponse(status_code=404, content={"detail": "Not found"})

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """
        SPA catch-all: serve index.html for any route not matched by the API.
        This enables client-side routing (React Router).
        """
        # Don't intercept API routes or docs
        if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
            return JSONResponse(status_code=404, content={"detail": "Not found"})

        # Try to serve the exact file if it exists (for .js, .css, .png, etc.).
        # Resolve first and refuse anything that escapes the frontend directory, so a
        # crafted path like "..%2f..%2f.env" cannot read outside frontend/dist.
        if full_path:
            candidate = (_frontend_dir / full_path).resolve()
            try:
                candidate.relative_to(_frontend_dir.resolve())
            except ValueError:
                return JSONResponse(status_code=404, content={"detail": "Not found"})
            if candidate.is_file():
                return FileResponse(str(candidate))

        # An asset-looking request that didn't resolve to a real file must 404 loudly.
        # Falling through to index.html here returns HTTP 200 with HTML where the
        # browser expected JavaScript, which renders a blank page and no useful error.
        if Path(full_path).suffix.lower() in _ASSET_SUFFIXES or full_path.startswith("assets/"):
            return JSONResponse(
                status_code=404, content={"detail": f"Asset not found: {full_path}"}
            )

        # Otherwise serve index.html (SPA fallback)
        index_path = _frontend_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))

        return JSONResponse(status_code=404, content={"detail": "Not found"})
