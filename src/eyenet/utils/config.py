from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    data = yaml.safe_load(text) if yaml is not None else parse_simple_yaml(text)
    data = data or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line: {raw_line}")
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if raw_value == "":
            value: Any = {}
            parent[key] = value
            stack.append((indent, value))
        else:
            parent[key] = parse_scalar(raw_value)
    return root


def parse_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


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
