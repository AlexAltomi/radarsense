"""Entry point: `uvicorn replay.main:app`. The cache and splits paths can be set with RADARSENSE_CACHE_DIR and
RADARSENSE_SPLITS."""
from replay.server import create_app

app = create_app()
