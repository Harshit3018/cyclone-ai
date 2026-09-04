"""
CYCLONE-AI Production Server Launcher.

Usage:
    python backend/run.py
    python backend/run.py --port 8000 --host 0.0.0.0
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def configure_logging(level: str = "INFO"):
    """Configure application-wide logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(description="CYCLONE-AI Server")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8000")))
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (use 1 for SQLite)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "info"))
    args = parser.parse_args()

    configure_logging(args.log_level)
    logger = logging.getLogger("cyclone_ai")

    logger.info("=" * 60)
    logger.info("  ██████╗██╗   ██╗ ██████╗██╗      ██████╗ ███╗   ██╗███████╗")
    logger.info(" ██╔════╝╚██╗ ██╔╝██╔════╝██║     ██╔═══██╗████╗  ██║██╔════╝")
    logger.info(" ██║      ╚████╔╝ ██║     ██║     ██║   ██║██╔██╗ ██║█████╗  ")
    logger.info(" ██║       ╚██╔╝  ██║     ██║     ██║   ██║██║╚██╗██║██╔══╝  ")
    logger.info(" ╚██████╗   ██║   ╚██████╗███████╗╚██████╔╝██║ ╚████║███████╗")
    logger.info("  ╚═════╝   ╚═╝    ╚═════╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝")
    logger.info("           AI-Powered Cyclone Intelligence Platform")
    logger.info("=" * 60)
    logger.info(f"  Host: {args.host}:{args.port}")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Reload: {args.reload}")
    logger.info(f"  Log Level: {args.log_level}")
    logger.info("=" * 60)

    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
