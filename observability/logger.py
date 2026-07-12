import logging
import os
import sys
from pathlib import Path


def _resolve_log_dir():
    candidates = []
    env_dir = os.getenv("IMAGE_PROCESS_LOG_DIR")
    if env_dir:
        candidates.append(env_dir)

    candidates.extend([
        "/app/logs",
        str(Path(__file__).resolve().parents[1] / "logs"),
        "/tmp/image-process-api/logs",
    ])

    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except (PermissionError, OSError):
            continue

    return "/tmp/image-process-api/logs"


log_dir = _resolve_log_dir()
log_file = os.path.join(log_dir, "app_performance.log")

handlers = [logging.StreamHandler(sys.stdout)]

try:
    handlers.append(logging.FileHandler(log_file))
except (PermissionError, OSError):
    handlers = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers,
    force=True,
)

logger = logging.getLogger("image-process-api")