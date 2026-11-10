"""ASGI entry point:  uvicorn app.main:app --host 0.0.0.0 --port 8000"""

import logging

from .api import create_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = create_app()
