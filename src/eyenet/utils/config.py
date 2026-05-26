from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    data = yaml.safe_load(text)
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data

def section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {key}")
    return value


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def attach_arg_defaults(parser, args):
    args._defaults = {action.dest: action.default for action in parser._actions if action.dest != "help"}
    return args


def cfg_arg(args, config: dict[str, Any], attr: str, dotted_key: str):
    current = getattr(args, attr)
    default = getattr(args, "_defaults", {}).get(attr, current)
    if current != default:
        return current
    return cfg_get(config, dotted_key, current)


def cfg_bool_with_disable_flag(
    args,
    config: dict[str, Any],
    disable_attr: str,
    dotted_key: str,
    default: bool,
) -> bool:
    """Resolve a boolean config value with a one-way CLI disable flag."""
    if getattr(args, disable_attr):
        return False
    return bool(cfg_get(config, dotted_key, default))
