"""Build prompts by loading templates and injecting context."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .logger_setup import get_logger
from .models import TaskNode

logger = get_logger("prompt")

_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "prompt_templates"


class PromptBuilder:
    """Load .txt templates and fill in variables for each phase."""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR

    def _load(self, name: str) -> str:
        path = self.template_dir / name
        if path.exists():
            return path.read_text(encoding="utf-8")
        logger.warning("Template not found: %s", path)
        return ""

    def system(self) -> str:
        return self._load("system.txt")

    def _read_files(self, paths: list[str], workspace: str) -> str:
        """Read context/data files and format them for the prompt."""
        if not paths:
            return ""
        parts = []
        for p in paths:
            full = Path(workspace) / p
            if full.exists():
                content = full.read_text(encoding="utf-8", errors="replace")
                if len(content) > 5000:
                    content = content[:5000] + "\n... [truncated]"
                parts.append(f"--- {p} ---\n{content}")
            else:
                parts.append(f"--- {p} --- (file not found)")
        return "\n".join(parts)

    def judge(self, task: TaskNode, workspace: str) -> str:
        tpl = self._load("judge.txt")

        # Build data input section from the task's DataPort
        data_section = ""
        if task.data_port.input_files:
            data_section = self._read_files(task.data_port.input_files, workspace)
        elif task.data_port.input_description:
            data_section = task.data_port.input_description
        else:
            data_section = "(no input data provided — if test data is needed, make 'prepare test data' a subtask)"

        context_section = ""
        if task.context_files:
            context_section = "参考文件：\n" + self._read_files(task.context_files, workspace)

        return tpl.format(
            task_description=task.description,
            data_input_section=data_section,
            context_section=context_section,
        )

    def execute(self, task: TaskNode, workspace: str) -> str:
        tpl = self._load("execute.txt")
        v = task.verification
        context_section = ""
        if task.context_files:
            context_section = "参考文件：\n" + self._read_files(task.context_files, workspace)
        return tpl.format(
            task_description=task.description,
            verification_description=v.description if v else "",
            expected_output=v.expected_output if v else "",
            verification_command=v.command if v else "",
            context_section=context_section,
        )

    def fix(self, task: TaskNode, error_info: str, workspace: str) -> str:
        tpl = self._load("fix.txt")
        v = task.verification
        context_section = ""
        if task.context_files:
            context_section = "参考文件：\n" + self._read_files(task.context_files, workspace)
        return tpl.format(
            task_description=task.description,
            verification_description=v.description if v else "",
            expected_output=v.expected_output if v else "",
            verification_command=v.command if v else "",
            error_info=error_info,
            context_section=context_section,
        )

    def backtrack(
        self, parent: TaskNode, failure_details: str, workspace: str,
    ) -> str:
        tpl = self._load("backtrack.txt")
        context_section = ""
        if parent.context_files:
            context_section = "参考文件：\n" + self._read_files(parent.context_files, workspace)
        return tpl.format(
            parent_description=parent.description,
            failure_details=failure_details,
            context_section=context_section,
        )

    def integrate(
        self, parent: TaskNode, children_summary: str, workspace: str,
    ) -> str:
        tpl = self._load("integrate.txt")
        context_section = ""
        if parent.context_files:
            context_section = "参考文件：\n" + self._read_files(parent.context_files, workspace)
        return tpl.format(
            parent_description=parent.description,
            children_summary=children_summary,
            context_section=context_section,
        )
