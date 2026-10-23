"""Shared contract models and schema validation utilities for Java <-> Python boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema


def validate_against_schema(payload: dict[str, Any], schema_path: str | Path) -> None:
    """Validate a dictionary payload against a JSON Schema Draft 2020-12 file."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    validator.validate(payload)
