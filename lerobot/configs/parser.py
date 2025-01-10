import inspect
import sys
from argparse import ArgumentError
from dataclasses import Field
from functools import wraps
from pathlib import Path
from typing import Sequence

import draccus

PATH_KEY = "path"
draccus.set_config_type("json")


def get_cli_overrides(field_name: str, args: Sequence[str] | None = None) -> list[str] | None:
    """Parses arguments from cli at a given nested attribute level.

    For example, supposing the main script was called with:
    python myscript.py --arg1=1 --arg2.subarg1=abc --arg2.subarg2=some/path

    If called during execution of myscript.py, get_cli_overrides("arg2") will return:
    ["--subarg1=abc" "--subarg2=some/path"]
    """
    if args is None:
        args = sys.argv[1:]
    attr_level_args = []
    detect_string = f"--{field_name}."
    exclude_strings = (f"--{field_name}.{draccus.CHOICE_TYPE_KEY}=", f"--{field_name}.{PATH_KEY}=")
    for arg in args:
        if arg.startswith(detect_string) and not arg.startswith(exclude_strings):
            denested_arg = f"--{arg.removeprefix(detect_string)}"
            attr_level_args.append(denested_arg)

    return attr_level_args


def parse_arg(arg_name: str, args: Sequence[str] | None = None) -> str | None:
    if args is None:
        args = sys.argv[1:]
    prefix = f"--{arg_name}="
    for arg in args:
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def get_path_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{PATH_KEY}", args)


def get_type_arg(field_name: str, args: Sequence[str] | None = None) -> str | None:
    return parse_arg(f"{field_name}.{draccus.CHOICE_TYPE_KEY}", args)


def preprocess_path_args(fields: Field | list[Field], args: Sequence[str] | None = None) -> list[str]:
    if isinstance(fields, Field):
        fields = [fields]

    filtered_args = args
    for field in fields:
        if get_path_arg(field.name, args):
            if get_type_arg(field.name, args):
                raise ArgumentError(
                    argument=None,
                    message=f"Cannot specify both --{field.name}.{PATH_KEY} and --{field.name}.{draccus.CHOICE_TYPE_KEY}",
                )
            filtered_args = [arg for arg in filtered_args if not arg.startswith(f"--{field.name}.")]

    return filtered_args


def wrap(config_path: Path | None = None, pathable_args: str | list[str] | None = None):
    """
    HACK: Similar to draccus.wrap but adds a preprocessing of CLI args in order to overload specific parts of
    the config from a config file at that particular nesting level.
    """

    def wrapper_outer(fn):
        @wraps(fn)
        def wrapper_inner(*args, **kwargs):
            argspec = inspect.getfullargspec(fn)
            argtype = argspec.annotations[argspec.args[0]]
            if len(args) > 0 and type(args[0]) is argtype:
                cfg = args[0]
                args = args[1:]
            else:
                cli_args = sys.argv[1:]
                get_path_fields = getattr(argtype, "__get_path_fields__", None)
                if get_path_fields:
                    path_fields = get_path_fields()
                    cli_args = preprocess_path_args(path_fields, cli_args)
                cfg = draccus.parse(config_class=argtype, config_path=config_path, args=cli_args)
            response = fn(cfg, *args, **kwargs)
            return response

        return wrapper_inner

    return wrapper_outer
