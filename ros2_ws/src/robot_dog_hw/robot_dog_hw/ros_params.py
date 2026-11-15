"""Helpers to declare and coerce ROS 2 parameters.

Parameters are declared with ``dynamic_typing``; the ``as_*`` helpers enforce the type and
raise ``ValueError`` naming the parameter.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional

from rcl_interfaces.msg import ParameterDescriptor


def declare(node, name: str, default: Any, description: str = '', read_only: bool = False):
    """Declare ``name`` with ``default``; read_only params cannot be changed at runtime."""
    desc = ParameterDescriptor(description=description, read_only=read_only,
                               dynamic_typing=True)
    return node.declare_parameter(name, default, desc)


def _num(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name}: expected a number, got {value!r}')
    return value


def as_float(value, name: str, finite: bool = True) -> float:
    v = float(_num(value, name))
    if finite and not math.isfinite(v):
        raise ValueError(f'{name}: must be finite, got {value!r}')
    return v


def as_int(value, name: str) -> int:
    v = _num(value, name)
    if isinstance(v, float):
        if not v.is_integer():
            raise ValueError(f'{name}: expected an integer, got {value!r}')
        v = int(v)
    return int(v)


def as_bool(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name}: expected true/false, got {value!r}')
    return value


def as_str(value, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{name}: expected a string, got {value!r}')
    return value


def _seq(value, name, n):
    if isinstance(value, (str, bytes)) or not hasattr(value, '__len__'):
        raise ValueError(f'{name}: expected a list, got {value!r}')
    seq = list(value)
    if n is not None and len(seq) != n:
        raise ValueError(f'{name}: expected {n} entries, got {len(seq)}')
    return seq


def as_float_list(value, name: str, n: Optional[int] = None) -> List[float]:
    return [as_float(v, f'{name}[{i}]') for i, v in enumerate(_seq(value, name, n))]


def as_int_list(value, name: str, n: Optional[int] = None) -> List[int]:
    return [as_int(v, f'{name}[{i}]') for i, v in enumerate(_seq(value, name, n))]


def as_bool_list(value, name: str, n: Optional[int] = None) -> List[bool]:
    return [as_bool(v, f'{name}[{i}]') for i, v in enumerate(_seq(value, name, n))]


def as_str_list(value, name: str, n: Optional[int] = None) -> List[str]:
    return [as_str(v, f'{name}[{i}]') for i, v in enumerate(_seq(value, name, n))]
