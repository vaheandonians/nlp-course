import types

from functools import wraps
from inspect import signature
from typing import get_args


def validate_args_not_none(*arg_names):

    # Sample:
    # @validate_args_not_none('text', 'another_arg')
    # def process_text(text, another_arg, optional_arg=None):

    def decorator(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            for arg_name in arg_names:
                if arg_name not in bound_arguments.arguments:
                    raise ValueError(f"[{str(func)}] '{arg_name}' is not part of the function signature.")

                if bound_arguments.arguments[arg_name] is None:
                    raise ValueError(f"[{str(func)}] Argument '{arg_name}' cannot be None.")

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_args(arg_types):

    # Sample:
    # @validate_args({'text': str, 'number': Union[int, float]})
    # def process_data(text, number):

    def decorator(func):
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_arguments = sig.bind(*args, **kwargs)
            bound_arguments.apply_defaults()

            for arg_name, arg_type in arg_types.items():
                if arg_name not in bound_arguments.arguments:
                    raise ValueError(f"[{str(func)}] '{arg_name}' is not part of the function signature.")
                actual_arg = bound_arguments.arguments[arg_name]
                if actual_arg is None:
                    raise ValueError(f"[{str(func)}] Argument '{arg_name}' cannot be None.")

                if isinstance(arg_type, types.UnionType):
                    expected_types = get_args(arg_type)
                else:
                    expected_types = (arg_type,)

                if not isinstance(actual_arg, expected_types):
                    expected_type_names = [t.__name__ for t in expected_types]
                    raise TypeError(f"[{str(func)}] Argument '{arg_name}' must be of type "
                                    f"{' or '.join(expected_type_names)}, but got {type(actual_arg).__name__}")

            return func(*args, **kwargs)

        return wrapper
    return decorator
