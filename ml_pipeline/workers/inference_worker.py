"""ML inference worker entry point."""

import signal
import sys
import time

from app.logging import configure_logging, get_logger
from app.config.settings import get_settings

logger = get_logger(__name__)


def _shutdown(signum: int, frame: object) -> None:
    logger.info("ml_worker_shutdown", signal=signum)
    sys.exit(0)


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    logger.info("ml_worker_started", environment=settings.app_env)
    while True:
        time.sleep(30)
        logger.debug("ml_worker_heartbeat")


if __name__ == "__main__":
    main()
