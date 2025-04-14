import logging

def init_logger(log_file_path: str) -> logging.Logger:
  """
  Initializes and returns a logger instance that writes logs to a file.

  This function sets up a logger named "logger" with DEBUG level, and attaches
  a FileHandler to log messages to the specified file. If the logger already has
  handlers attached, it avoids adding duplicate handlers.

  Args:
      log_file_path (str): Path to the log file where logs will be written.

  Returns:
      logging.Logger: Configured logger instance.

  Example usage:
      logger = init_logger("logs/train.log")
      logger.info("Training started.")
  """
  logger = logging.getLogger("logger")
  logger.setLevel(logging.DEBUG)

  # Avoid adding multiple handlers if logger already has one
  if not logger.handlers:
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

  return logger
