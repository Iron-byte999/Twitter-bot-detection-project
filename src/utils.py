import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def data_path(filename: str) -> Path:
    """Return a path inside the project data directory."""
    return project_root() / "data" / filename


def reports_path(filename: str) -> Path:
    """Return a report path and create the reports directory if needed."""
    path = project_root() / "reports" / filename
    path.parent.mkdir(exist_ok=True)
    return path


def models_path(filename: str) -> Path:
    """Return a model path and create the models directory if needed."""
    path = project_root() / "models" / filename
    path.parent.mkdir(exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
