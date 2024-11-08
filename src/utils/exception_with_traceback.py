import traceback
from functools import wraps


def exception_with_traceback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise Exception(f"{e}\n{traceback.format_exc()}") from e
    return wrapper
