"""Task tree and API call record persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .logger_setup import get_logger
from .models import TaskTree

logger = get_logger("persistence")


class Persistence:
    """Save / load task tree and API call records to the workspace."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._calls_dir = self.workspace / "api_calls"
        self._calls_dir.mkdir(exist_ok=True)
        self._call_counter = 0

    def save_tree(self, tree: TaskTree) -> None:
        path = self.workspace / "task_tree.json"
        path.write_text(
            json.dumps(tree.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_tree(self) -> Optional[TaskTree]:
        path = self.workspace / "task_tree.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return TaskTree.from_dict(data)

    def save_api_call(self, record: dict) -> str:
        """Save a single API call record. Returns the call_id."""
        self._call_counter += 1
        call_id = f"call_{self._call_counter:04d}"
        record["call_id"] = call_id
        path = self._calls_dir / f"{call_id}.json"
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return call_id

    def save_report(self, report: dict) -> None:
        path = self.workspace / "evaluation_report.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_report(self) -> Optional[dict]:
        path = self.workspace / "evaluation_report.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
