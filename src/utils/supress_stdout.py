import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    try:
        sys.stdout = open(os.devnull, 'w')
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
