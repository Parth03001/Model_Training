"""Entry point — run with: python main.py (from datavision-platform/backend/)"""

import uvicorn
from app.config import settings
from app.main import app  # noqa: F401 — registers all routes

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
