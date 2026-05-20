from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _to_jsonable(payload: Any) -> Any:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(k): _to_jsonable(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_to_jsonable(v) for v in payload]
    return payload


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def ensure_run_dir(results_root: Path | str, run_id: str) -> Path:
    run_dir = Path(results_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path | str, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = _to_jsonable(payload)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_manifest(run_dir: Path | str, manifest: Any) -> Path:
    return write_json(Path(run_dir) / "manifest.json", manifest)


def write_result(run_dir: Path | str, result: Any) -> Path:
    results_path = Path(run_dir) / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_jsonable(result)
    with results_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, sort_keys=True) + "\n")
    return results_path
