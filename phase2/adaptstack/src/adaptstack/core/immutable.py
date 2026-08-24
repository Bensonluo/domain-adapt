"""Deeply immutable JSON-compatible values for public contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_value(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("immutable contract values must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("immutable contract mapping keys must be non-empty strings")
            frozen[key] = freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    raise TypeError(f"unsupported immutable contract value type: {type(value).__name__}")


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen = freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the signature
        raise TypeError("value must be a mapping")
    return frozen


def plain_value(value: Any) -> Any:
    """Convert frozen values into fresh JSON-compatible containers."""

    if isinstance(value, Mapping):
        return {key: plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_value(item) for item in value]
    return value
