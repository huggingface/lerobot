# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that documented arguments match the real signature.

Adapted from the core of `transformers/utils/check_docstrings.py`. The parts of that file bound to
transformers internals — the `@auto_docstring` decorator system, modular-file propagation, `ModelArgs`,
GitPython — are deliberately not ported.

What this enforces, for every public object in `MODULES_TO_CHECK`:

- every parameter in the signature has an `Args:` entry, in signature order;
- no `Args:` entry names a parameter that does not exist;
- the `*optional*, defaults to `X`` clause matches the real default.

That last one is why the clause is not decorative. See docs/source/writing_docstrings.mdx.

Check, as CI does:

```bash
python utils/check_docstrings.py
```

Rewrite the `Args:` blocks to match the signatures, inserting `<fill_docstring>` placeholders for
parameters that are missing entirely:

```bash
python utils/check_docstrings.py --fix_and_overwrite
```

`MODULES_TO_CHECK` is the ratchet: add a module once its docstrings are converted.
"""

import argparse
import ast
import enum
import importlib
import inspect
import operator as op
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any

PATH_TO_REPO = Path(__file__).resolve().parent.parent
PATH_TO_LEROBOT = PATH_TO_REPO / "src" / "lerobot"

# Modules whose public objects are checked. Add a module here once its docstrings follow the standard.
MODULES_TO_CHECK = [
    "lerobot.robots",
]

# Objects that do not yet follow the standard, so the check can be green from day one. Removing an entry
# and running `--fix_and_overwrite` is how a module gets converted.
#
# Every entry below has a bare `Attributes:` section, which this checker reads as an argument section (the
# same aliasing doc-builder does) and therefore compares against the signature. They are converted in the
# docstring PR that follows this one, which empties this set.
OBJECTS_TO_IGNORE: set[str] = {
    "ChannelFactoryInitialize",
    "EarthRoverMiniPlus",
    "EarthRoverMiniPlusConfig",
    "EEBoundsAndSafety",
    "EEReferenceAndDelta",
    "ForwardKinematicsJointsToEEAction",
    "ForwardKinematicsJointsToEEObservation",
    "GripperVelocityToJoint",
    "InverseKinematicsEEToJoints",
    "Robot",
}

OPTIONAL_KEYWORD = "*optional*"

_re_args = re.compile(r"^\s*(Args?|Arguments?|Attributes?|Params?|Parameters?):\s*$")
_re_parse_arg = re.compile(r"^(\s*)(\S+)\s+\((.+)\)(?:\:|$)")
_re_parse_description = re.compile(r"\*optional\*, defaults to (.*)$")

MATH_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}


def find_indent(line: str) -> int:
    """Return the number of spaces a line is indented by.

    Args:
        line (`str`):
            The line to measure.

    Returns:
        `int`: The indentation width.
    """
    search = re.search(r"^(\s*)(?:\S|$)", line)
    return 0 if search is None else len(search.groups()[0])


def is_dataclass_factory_default(default: Any) -> bool:
    """Whether a signature default came from a dataclass `field(default_factory=...)`.

    `inspect.signature` renders those as a `<factory>` sentinel, which must not be written into a
    docstring as a literal default.

    Args:
        default (`Any`):
            The default value taken from the signature.

    Returns:
        `bool`: `True` for the factory sentinel.
    """
    return repr(default) == "<factory>"


def stringify_default(default: Any) -> str:
    """Render a default value the way a docstring should show it.

    Args:
        default (`Any`):
            The default value to process.

    Returns:
        `str`: Numbers are left bare, everything else is wrapped in backticks.
    """
    if isinstance(default, bool):
        # Must precede the int check: a bool passes isinstance(x, int).
        return f"`{default}`"
    elif isinstance(default, enum.Enum):
        # Must also precede the int check: an IntEnum passes isinstance(x, int).
        return f"`{str(default)}`"
    elif isinstance(default, int):
        return str(default)
    elif isinstance(default, float):
        result = str(default)
        return str(round(default, 2)) if len(result) > 6 else result
    elif isinstance(default, str):
        return str(default) if default.isnumeric() else f'`"{default}"`'
    elif isinstance(default, type):
        return f"`{default.__name__}`"
    else:
        return f"`{default}`"


def eval_node(node):
    """Evaluate one node of a arithmetic-only AST.

    Args:
        node (`ast.AST`):
            The node to evaluate.

    Returns:
        `float | int | complex`: The node's value.

    Raises:
        TypeError: If the node is not a number or a supported arithmetic operation.
    """
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
        return node.value
    elif isinstance(node, ast.BinOp):
        return MATH_OPERATORS[type(node.op)](eval_node(node.left), eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):
        return MATH_OPERATORS[type(node.op)](eval_node(node.operand))
    else:
        raise TypeError(node)


def eval_math_expression(expression: str) -> float | int | None:
    """Safely evaluate an arithmetic expression found in a docstring.

    Docstrings often document a default as an expression (`1 / 255` is the classic), which should be left
    alone rather than replaced by its computed value.

    Args:
        expression (`str`):
            The expression to evaluate.

    Returns:
        `float | int | None`: The value, or `None` if it is not a plain arithmetic expression.
    """
    try:
        return eval_node(ast.parse(expression, mode="eval").body)
    except (TypeError, SyntaxError, KeyError, ZeroDivisionError):
        return None


def replace_default_in_arg_description(description: str, default: Any) -> str:
    """Rewrite the `*optional*, defaults to X` clause of one argument description.

    Args:
        description (`str`):
            The argument description from the docstring, without the name.
        default (`Any`):
            The real default from the signature, or `inspect._empty` if the argument is required.

    Returns:
        `str`: The description with its optional/default clause matching the signature.
    """
    # Plenty of docstrings use `optional` or **optional** instead of *optional*.
    description = description.replace("`optional`", OPTIONAL_KEYWORD)
    description = description.replace("**optional**", OPTIONAL_KEYWORD)

    if default is inspect._empty:
        # Required: the description must not claim otherwise.
        idx = description.find(OPTIONAL_KEYWORD)
        if idx != -1:
            description = description[:idx].rstrip().removesuffix(",").rstrip()
    elif default is None or is_dataclass_factory_default(default):
        # A `None` default is not spelled out, and a `default_factory` has no literal value to show.
        idx = description.find(OPTIONAL_KEYWORD)
        if idx == -1:
            description = f"{description}, {OPTIONAL_KEYWORD}"
        elif re.search(r"defaults to `?None`?", description) is not None:
            description = description[: idx + len(OPTIONAL_KEYWORD)]
    else:
        str_default = None
        documented_match = re.search("defaults to `?(.*?)(?:`|$)", description)
        if isinstance(default, (int, float)) and documented_match is not None:
            documented = documented_match.groups()[0]
            if default == eval_math_expression(documented):
                try:
                    # Directly convertible means it was a plain literal.
                    str_default = str(type(default)(documented))
                except (TypeError, ValueError):
                    # Otherwise it was an expression; keep it as written.
                    str_default = f"`{documented}`"

        if str_default is None:
            str_default = stringify_default(default)

        if OPTIONAL_KEYWORD not in description:
            description = f"{description}, {OPTIONAL_KEYWORD}, defaults to {str_default}"
        elif _re_parse_description.search(description) is None:
            idx = description.find(OPTIONAL_KEYWORD)
            description = f"{description[: idx + len(OPTIONAL_KEYWORD)]}, defaults to {str_default}"
        else:
            description = _re_parse_description.sub(f"*optional*, defaults to {str_default}", description)

    return description


def get_default_description(arg: inspect.Parameter) -> str:
    """Build the parenthesised type-and-default part for an undocumented parameter.

    Args:
        arg (`inspect.Parameter`):
            The parameter to describe.

    Returns:
        `str`: Something like ``` `int`, *optional*, defaults to 3 ```.
    """
    if arg.annotation is inspect._empty:
        arg_type = "<fill_type>"
    elif hasattr(arg.annotation, "__name__"):
        arg_type = arg.annotation.__name__
    else:
        arg_type = str(arg.annotation)

    if arg.default is inspect._empty:
        return f"`{arg_type}`"
    elif arg.default is None or is_dataclass_factory_default(arg.default):
        return f"`{arg_type}`, {OPTIONAL_KEYWORD}"
    else:
        return f"`{arg_type}`, {OPTIONAL_KEYWORD}, defaults to {stringify_default(arg.default)}"


def find_source_file(obj: Any) -> Path:
    """Locate the file an object is defined in.

    Args:
        obj (`Any`):
            The object to locate.

    Returns:
        `Path`: The source file.
    """
    obj_file = PATH_TO_LEROBOT
    for part in obj.__module__.split(".")[1:]:
        obj_file = obj_file / part
    return obj_file.with_suffix(".py")


def match_docstring_with_signature(obj: Any) -> tuple[str, str] | None:
    """Compare an object's documented arguments against its signature.

    Dataclasses need no special handling: `inspect.signature` resolves the generated `__init__`, inherited
    fields included, which is exactly the set a reader sees on the rendered page.

    Args:
        obj (`Any`):
            The class or function to check.

    Returns:
        `tuple[str, str] | None`: The current `Args:` block and the one matching the signature, or `None`
        when there is nothing to compare — no docstring, no documented arguments, or an unsupported
        signature.
    """
    if not getattr(obj, "__doc__", None):
        return None

    try:
        source, _ = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        source = []

    idx = 0
    while idx < len(source) and '"""' not in source[idx]:
        idx += 1

    ignore_order = False
    if idx < len(source) and idx > 0:
        line_before_docstring = source[idx - 1]
        if re.search(r"^\s*#\s*no-format\s*$", line_before_docstring):
            return None
        elif re.search(r"^\s*#\s*ignore-order\s*$", line_before_docstring):
            ignore_order = True

    try:
        signature = inspect.signature(obj).parameters
    except (ValueError, TypeError):
        return None

    obj_doc_lines = obj.__doc__.split("\n")
    idx = 0
    while idx < len(obj_doc_lines) and _re_args.search(obj_doc_lines[idx]) is None:
        idx += 1
    if idx == len(obj_doc_lines):
        # No arguments documented; coverage is interrogate's job, not this check's.
        return None

    if "kwargs" in signature and signature["kwargs"].annotation != inspect._empty:
        # Typed **kwargs are not introspectable in a useful way here.
        return None

    indent = find_indent(obj_doc_lines[idx])
    arguments: dict[str, Any] = {}
    current_arg = None
    idx += 1
    start_idx = idx
    # Consume until a non-empty line returns to the section's own indent, or the docstring ends.
    while idx < len(obj_doc_lines) and (
        len(obj_doc_lines[idx].strip()) == 0 or find_indent(obj_doc_lines[idx]) > indent
    ):
        if find_indent(obj_doc_lines[idx]) == indent + 4:
            re_search_arg = _re_parse_arg.search(obj_doc_lines[idx])
            if re_search_arg is not None:
                _, name, description = re_search_arg.groups()
                current_arg = name
                if name in signature:
                    default = signature[name].default
                    if signature[name].kind is inspect._ParameterKind.VAR_KEYWORD:
                        default = None
                    new_description = replace_default_in_arg_description(description, default)
                else:
                    new_description = description
                arguments[current_arg] = [
                    _re_parse_arg.sub(rf"\1\2 ({new_description}):", obj_doc_lines[idx])
                ]
        elif current_arg is not None:
            arguments[current_arg].append(obj_doc_lines[idx])
        idx += 1

    # Walk back over the trailing blank lines we consumed.
    idx -= 1
    if current_arg:
        while len(obj_doc_lines[idx].strip()) == 0:
            arguments[current_arg] = arguments[current_arg][:-1]
            idx -= 1
    idx += 1

    old_doc_arg = "\n".join(obj_doc_lines[start_idx:idx])

    old_arguments = list(arguments.keys())
    arguments = {name: "\n".join(doc) for name, doc in arguments.items()}
    for name in set(signature.keys()) - set(arguments.keys()):
        arg = signature[name]
        # Private parameters and *args/**kwargs are only documented if the author chose to.
        if name.startswith("_") or arg.kind in [
            inspect._ParameterKind.VAR_KEYWORD,
            inspect._ParameterKind.VAR_POSITIONAL,
        ]:
            arguments[name] = ""
        else:
            arguments[name] = (
                " " * (indent + 4) + f"{name} ({get_default_description(arg)}): <fill_docstring>"
            )

    if ignore_order:
        new_param_docs = [arguments[name] for name in old_arguments if name in signature]
        missing = set(signature.keys()) - set(old_arguments)
        new_param_docs.extend([arguments[name] for name in missing if len(arguments[name]) > 0])
    else:
        new_param_docs = [arguments[name] for name in signature if len(arguments[name]) > 0]

    return old_doc_arg, "\n".join(new_param_docs)


def fix_docstring(obj: Any, old_doc_args: str, new_doc_args: str) -> None:
    """Rewrite an object's `Args:` block in its source file.

    Args:
        obj (`Any`):
            The object whose docstring is being fixed.
        old_doc_args (`str`):
            The current `Args:` block, as returned by [`match_docstring_with_signature`].
        new_doc_args (`str`):
            The replacement block, as returned by [`match_docstring_with_signature`].

    Raises:
        ValueError: If the block found in the source does not match the one parsed from `__doc__`, which
            means the boundaries were identified wrongly and rewriting would corrupt the file.
    """
    source, line_number = inspect.getsourcelines(obj)

    idx = 0
    while idx < len(source) and _re_args.search(source[idx]) is None:
        idx += 1
    if idx == len(source):
        # Inherited docstring: do not rewrite it on the child.
        return

    indent = find_indent(source[idx])
    idx += 1
    start_idx = idx
    while idx < len(source) and (len(source[idx].strip()) == 0 or find_indent(source[idx]) > indent):
        idx += 1
    idx -= 1
    while len(source[idx].strip()) == 0:
        idx -= 1
    idx += 1

    # `old_doc_args` comes from `__doc__`, whose indentation differs from the raw source lines.
    source_args_as_str = "".join(source[start_idx:idx])
    if inspect.cleandoc(source_args_as_str) != inspect.cleandoc(old_doc_args):
        raise ValueError(
            f"Cannot fix the docstring of {obj.__name__} in {find_source_file(obj)}: the argument section "
            f"in the source does not match the one parsed from __doc__, so the block boundaries are "
            f"wrong and rewriting it would corrupt the file.\n\n"
            f"Parsed:\n{old_doc_args!r}\n\nFound in source:\n{source_args_as_str.rstrip()!r}\n"
        )

    obj_file = find_source_file(obj)
    lines = obj_file.read_text(encoding="utf-8").split("\n")
    # `new_doc_args` is built from `__doc__`, and Python keeps every line after the first at its exact
    # source indentation, so the block is already correctly indented for the file. transformers re-indents
    # here because its docstrings are often assembled by decorators and no longer match the source.
    lines = lines[: line_number + start_idx - 1] + [new_doc_args] + lines[line_number + idx - 1 :]

    print(f"Fixing the docstring of {obj.__name__} in {obj_file}.")
    obj_file.write_text("\n".join(lines), encoding="utf-8")


def iter_objects_to_check(module_name: str):
    """Yield the public classes and functions defined in a package.

    Args:
        module_name (`str`):
            An importable package name, e.g. `"lerobot.robots"`.

    Yields:
        `Any`: Each public class or function whose `__module__` is inside the package, deduplicated so
        that aliases (`SO101Follower = SOFollower`) are visited once.
    """
    package = importlib.import_module(module_name)
    module_names = [module_name]
    if hasattr(package, "__path__"):
        module_names += [
            name for _, name, _ in pkgutil.walk_packages(package.__path__, prefix=f"{module_name}.")
        ]

    seen = set()
    for name in module_names:
        try:
            module = importlib.import_module(name)
        except Exception as error:  # An optional extra is missing; not this check's problem.
            print(f"Skipping {name}: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        for attr_name, obj in vars(module).items():
            if attr_name.startswith("_") or not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            if not getattr(obj, "__module__", "").startswith(module_name):
                continue
            key = f"{obj.__module__}.{obj.__qualname__}"
            if key in seen or obj.__qualname__ in OBJECTS_TO_IGNORE or key in OBJECTS_TO_IGNORE:
                continue
            seen.add(key)
            yield obj


def check_docstrings(overwrite: bool = False) -> list[str]:
    """Check every object in `MODULES_TO_CHECK`.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to rewrite mismatched `Args:` blocks in place.

    Returns:
        `list[str]`: The names of objects whose documented arguments do not match their signature. Empty
        when everything is consistent.
    """
    failures = []
    hard_failures = []
    for module_name in MODULES_TO_CHECK:
        for obj in iter_objects_to_check(module_name):
            try:
                result = match_docstring_with_signature(obj)
            except Exception as error:
                hard_failures.append(f"{obj.__qualname__}: {type(error).__name__}: {error}")
                continue
            if result is None:
                continue
            old_doc, new_doc = result
            if old_doc == new_doc:
                continue
            if overwrite:
                fix_docstring(obj, old_doc, new_doc)
            else:
                failures.append(f"{obj.__module__}.{obj.__qualname__}")

    if hard_failures:
        print("The following objects could not be processed:", file=sys.stderr)
        for failure in hard_failures:
            print(f"- {failure}", file=sys.stderr)
    return failures


def main() -> int:
    """Run the check.

    Returns:
        `int`: `0` when every documented argument matches its signature, `1` otherwise.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    failures = check_docstrings(overwrite=args.fix_and_overwrite)
    if failures:
        print(
            "The docstrings of the following objects do not match their signature. Run "
            "`make fix-docstrings` to rewrite them, then fill in any `<fill_docstring>` placeholders:",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
