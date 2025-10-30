# # logger.py
# import logging
# import os
# from datetime import datetime

# def setup_logger(module_name: str) -> logging.Logger:
#     """
#     Sets up a logger for the given module name.
#     Logs will be saved to a file named {module_name}_YYYY-MM-DD_HH-MM-SS.log
#     and also printed to console.
#     """

#     # Create logs directory if it doesn’t exist
#     log_dir = "logs"
#     os.makedirs(log_dir, exist_ok=True)

#     # Construct dynamic log file name
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_file = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

#     # Define logging format
#     log_format = "%(asctime)s - %(levelname)s - %(message)s"

#     # Configure root logger
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format=log_format,
#         handlers=[
#             logging.FileHandler(log_file, encoding='utf-8'),
#             logging.StreamHandler()
#         ]
#     )

#     # Return module-specific logger
#     logger = logging.getLogger(module_name)
#     logger.debug(f"Logger initialized for {module_name}. Log file: {log_file}")
#     return logger



# logger.py
import logging
import os
import warnings
import glob
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List

def _purge_old_timestamped_logs(module_name: str, log_dir: str, keep: int) -> List[str]:
    """
    Delete older timestamped log files for module_name in log_dir,
    keeping only the newest `keep` files. Returns a list of deleted filenames.
    Only affects files that match the pattern: {module_name}_*.log
    Does NOT touch {module_name}.rotating.log
    """
    pattern = os.path.join(log_dir, f"{module_name}_*.log")
    files = glob.glob(pattern)

    # Exclude the rotating log (if by chance it matches)
    rotating_name = os.path.join(log_dir, f"{module_name}.rotating.log")
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(rotating_name)]

    if not files:
        return []

    # Sort by modification time (oldest first)
    files.sort(key=lambda p: os.path.getmtime(p))

    # If we have more than `keep`, delete the oldest ones
    deleted = []
    if len(files) > keep:
        to_delete = files[:-keep]  # oldest files
        for f in to_delete:
            try:
                os.remove(f)
                deleted.append(f)
            except Exception:
                # best-effort deletion; if it fails, continue
                pass
    return deleted


def setup_logger(module_name: str,
                 log_dir: str = "logs",
                 level: int = logging.DEBUG,
                 when: str = "midnight",
                 backup_count: int = 7,
                 keep: int = 10) -> logging.Logger:
    """
    Create and return a configured logger.

    - Logs written to logs/{module_name}_YYYY-MM-DD_HH-MM-SS.log (one per session)
    - Also keeps a rotating log at logs/{module_name}.rotating.log (TimedRotatingFileHandler)
    - Console logging enabled.
    - Automatically keeps only the `keep` most recent timestamped log files and deletes older ones.
      (Default keep=10)
    - Suppresses noisy Streamlit runtime warning about missing ScriptRunContext
      when running in non-streamlit mode.
    """

    os.makedirs(log_dir, exist_ok=True)

    # Primary timestamped filename (one-shot per session)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

    # Basic formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # Acquire module logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # If handlers already exist, still attempt to purge old logs (to enforce keep limit)
    if logger.handlers:
        try:
            deleted = _purge_old_timestamped_logs(module_name, log_dir, keep)
            if deleted:
                # Because handlers already present, logger.debug will go to existing handlers
                logger.debug(f"Purged {len(deleted)} old log(s): {deleted}")
        except Exception:
            # don't fail on cleanup
            pass
        logger.setLevel(level)
        return logger

    # File handler (timestamped file)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Timed rotating handler (also keeps recent rotations) for long-running processes
    rotate_filename = os.path.join(log_dir, f"{module_name}.rotating.log")
    timed_handler = TimedRotatingFileHandler(rotate_filename, when=when, backupCount=backup_count, utc=False)
    timed_handler.setLevel(level)
    timed_handler.setFormatter(formatter)
    logger.addHandler(timed_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # default console verbosity
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Reduce verbosity of some third-party loggers that often flood output
    try:
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("botocore").setLevel(logging.WARNING)
        logging.getLogger("boto3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("streamlit").setLevel(logging.WARNING)
    except Exception:
        pass

    # Suppress the specific noisy Streamlit warning when running script directly.
    warnings.filterwarnings(
        "ignore",
        message="Thread '.*': missing ScriptRunContext! This warning can be ignored when running in bare mode."
    )
    warnings.filterwarnings(
        "ignore",
        message="Session state does not function when running a script without `streamlit run`"
    )

    # After handlers are installed, purge older timestamped logs (keep only `keep`)
    try:
        deleted_files = _purge_old_timestamped_logs(module_name, log_dir, keep)
        if deleted_files:
            logger.debug(f"Purged {len(deleted_files)} old log(s): {deleted_files}")
    except Exception:
        # ignore cleanup errors
        pass

    logger.debug(f"Logger initialized for {module_name}. Log file: {log_filename}")
    return logger
