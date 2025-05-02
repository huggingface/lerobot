import logging
import logging.handlers
import os
import time


def setup_logging(prefix: str, info_bracket: str):
    """Sets up logging"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Delete any existing prefix_* log files
    for old_log_file in os.listdir("logs"):
        if old_log_file.startswith(prefix) and old_log_file.endswith(".log"):
            try:
                os.remove(os.path.join("logs", old_log_file))
                print(f"Deleted old log file: {old_log_file}")
            except Exception as e:
                print(f"Failed to delete old log file {old_log_file}: {e}")

    # Set up logging with both console and file output
    logger = logging.getLogger(prefix)
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False

    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)

    # File handler - creates a new log file for each run
    file_handler = logging.handlers.RotatingFileHandler(
        f"logs/policy_server_{int(time.time())}.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(
        logging.Formatter(
            f"%(asctime)s.%(msecs)03d [{info_bracket}] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    return logger
