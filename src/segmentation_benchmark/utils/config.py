"""YAML-based configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping

import yaml


def load_config(config_path: Path | str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Configuration root must be a mapping, found {type(data)!r}")
    return dict(data)


def merge_dict(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries and return the merged result."""
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], MutableMapping) and isinstance(value, Mapping):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def instantiate_from_config(
    name: str,
    registry: Mapping[str, Callable[..., Any]],
    config: Mapping[str, Any] | None = None,
    required_keys: Iterable[str] | None = None,
    **override_kwargs: Any,
) -> Any:
    """Instantiate an object from a registry using configuration values."""
    if name not in registry:
        available = ", ".join(sorted(registry)) or "<empty>"
        raise KeyError(f"Unknown registry entry '{name}'. Available: {available}")
    config = dict(config or {})
    if required_keys is not None:
        missing = [key for key in required_keys if key not in config and key not in override_kwargs]
        if missing:
            raise KeyError(f"Missing required config keys for '{name}': {missing}")
    kwargs = {**config, **override_kwargs}
    return registry[name](**kwargs)
