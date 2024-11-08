import functools
import time

from loguru import logger


def retry_on_exception(retries=3, delay=1):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    logger.error(f"Attempt {attempt} failed with error: {e}. Retrying in {delay} seconds...")
                    time.sleep(delay)
            logger.error(f"All {retries} attempts failed.")
            return None
        return wrapper
    return decorator_retry
