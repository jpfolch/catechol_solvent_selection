import argparse
from typing import TypeVar

T = TypeVar("T")
T_ = TypeVar("T")


def map_str_to_bool(s: T_) -> T_ | bool:
    if not isinstance(s, str):
        return s
    if s.lower() in ["true", "false"]:
        return s.lower() == "true"
    return s


def map_str_to_type(s: T_, type: type[T]) -> T_ | T:
    if not isinstance(s, str):
        return s
    try:
        out = type(s)
    except ValueError:
        out = s
    return out


class StoreDict(argparse.Action):
    """Custom action to support passing kwargs.

    https://stackoverflow.com/a/11762020"""

    def __call__(self, parser, namespace, values, option_string=None):
        # Create or retrieve an existing dictionary from the namespace.
        kwargs_dict = {}
        # Allow values to be passed as a list (supporting multiple key-value pairs)
        for value in values:
            try:
                key, _, val = value.partition("=")
                val = map_str_to_bool(val)
                val = map_str_to_type(val, int)
                val = map_str_to_type(val, float)
            except ValueError:
                message = f"Value '{value}' is not in key=value format"
                raise argparse.ArgumentError(self, message)
            kwargs_dict[key] = val
        setattr(namespace, self.dest, kwargs_dict)
