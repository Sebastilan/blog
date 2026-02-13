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

    # ── Formatting helpers ──

    def _format_execution_plan(self, task: TaskNode) -> str:
        """Format execution_plan as a prompt section."""
        if not task.execution_plan:
            return ""
        steps = "\n".join(task.execution_plan)
        return f"执行计划（请严格按此步骤执行）：\n{steps}"

    def _format_interface(self, task: TaskNode) -> str:
        """Format interface definition as a prompt section."""
        if not task.interface:
            return ""
        parts = ["接口定义（你的实现必须严格符合此接口）："]
        for key, val in task.interface.items():
            parts.append(f"- {key}: {val}")
        return "\n".join(parts)

    def _format_interface_contract(self, task: TaskNode) -> str:
        """Format interface_contract as a prompt section."""
        if not task.interface_contract:
            return ""
        return f"接口契约（所有子任务必须遵守的数据格式约定）：\n{task.interface_contract}"

    def _format_error_history(self, task: TaskNode) -> str:
        """Format full error history (not just last error) so agent avoids repeating mistakes."""
        if len(task.error_log) <= 1:
            return ""
        parts = ["之前的错误历史（请避免重复相同的错误）："]
        for i, err in enumerate(task.error_log[:-1], 1):
            parts.append(f"第 {i} 次尝试失败：{err[:500]}")
        return "\n".join(parts)

    def _format_parent_context(self, task: TaskNode, tree=None) -> str:
        """Format parent context info for child tasks during judge phase."""
        parts = []
        if task.interface_contract:
            parts.append(f"接口契约（本任务必须遵守的数据格式约定）：\n{task.interface_contract}")
        if task.parent_id and tree:
            parent = tree.get_node(task.parent_id)
            if parent:
                parts.append(f"父任务：{parent.description}")
                if parent.interface_contract:
                    parts.append(f"父任务接口契约：\n{parent.interface_contract}")
        if not parts:
            return ""
        return "\n\n".join(parts)

    def _format_ancestry_chain(self, task: TaskNode, tree=None) -> str:
        """Gap 3: Build an ancestry chain so deep tasks know their global context.

        Example output:
          全局目标：实现一个支持 +−×÷ 和括号的计算器
            └── 父任务：将计算器拆分为 tokenizer 和 evaluator
                └── 当前任务：写 tokenizer.py ...
        """
        if not tree or not task.parent_id:
            return ""

        # Walk up the tree to collect ancestors
        ancestors = []
        node = task
        while node.parent_id:
            parent = tree.get_node(node.parent_id)
            if not parent:
                break
            ancestors.append(parent.description)
            node = parent

        if not ancestors:
            return ""

        ancestors.reverse()  # root first
        parts = ["任务上下文（从全局目标到当前任务的链路）："]
        for i, desc in enumerate(ancestors):
            indent = "  " * i
            if i == 0:
                parts.append(f"{indent}全局目标：{desc}")
            else:
                parts.append(f"{indent}└── 父任务：{desc}")
        parts.append(f"{'  ' * len(ancestors)}└── 当前任务：{task.description}")

        # Also include interface_contract if present
        if task.interface_contract:
            parts.append(f"\n接口契约：\n{task.interface_contract}")

        return "\n".join(parts)

    # ── Phase prompts ──

    def judge(self, task: TaskNode, workspace: str, tree=None) -> str:
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

        parent_context_section = self._format_parent_context(task, tree)

        return tpl.format(
            task_description=task.description,
            data_input_section=data_section,
            context_section=context_section,
            parent_context_section=parent_context_section,
        )

    def execute(self, task: TaskNode, workspace: str, tree=None) -> str:
        tpl = self._load("execute.txt")
        v = task.verification
        context_section = ""
        if task.context_files:
            context_section = "参考文件：\n" + self._read_files(task.context_files, workspace)

        execution_plan_section = self._format_execution_plan(task)
        interface_section = self._format_interface(task)
        # Gap 3: Inject ancestry chain so agent knows global context
        ancestry_section = self._format_ancestry_chain(task, tree)

        return tpl.format(
            task_description=task.description,
            ancestry_section=ancestry_section,
            execution_plan_section=execution_plan_section,
            interface_section=interface_section,
            verification_description=v.description if v else "",
            expected_output=v.expected_output if v else "",
            verification_command=v.command if v else "",
            context_section=context_section,
        )

    def fix(self, task: TaskNode, error_info: str, workspace: str, tree=None) -> str:
        tpl = self._load("fix.txt")
        v = task.verification
        context_section = ""
        if task.context_files:
            context_section = "参考文件：\n" + self._read_files(task.context_files, workspace)

        interface_section = self._format_interface(task)
        error_history_section = self._format_error_history(task)
        # Gap 3: Inject ancestry chain
        ancestry_section = self._format_ancestry_chain(task, tree)

        return tpl.format(
            task_description=task.description,
            ancestry_section=ancestry_section,
            interface_section=interface_section,
            verification_description=v.description if v else "",
            expected_output=v.expected_output if v else "",
            verification_command=v.command if v else "",
            error_info=error_info,
            error_history_section=error_history_section,
            context_section=context_section,
        )

    def backtrack(
        self, parent: TaskNode, failure_details: str, workspace: str,
    ) -> str:
        tpl = self._load("backtrack.txt")
        context_section = ""
        if parent.context_files:
            context_section = "参考文件：\n" + self._read_files(parent.context_files, workspace)

        interface_contract_section = ""
        if parent.interface_contract:
            interface_contract_section = (
                f"上次的接口契约（可参考或修改）：\n{parent.interface_contract}"
            )

        return tpl.format(
            parent_description=parent.description,
            failure_details=failure_details,
            interface_contract_section=interface_contract_section,
            context_section=context_section,
        )

    def integrate(
        self, parent: TaskNode, children_summary: str, workspace: str,
    ) -> str:
        tpl = self._load("integrate.txt")
        context_section = ""
        if parent.context_files:
            context_section = "参考文件：\n" + self._read_files(parent.context_files, workspace)

        interface_contract_section = self._format_interface_contract(parent)

        return tpl.format(
            parent_description=parent.description,
            children_summary=children_summary,
            interface_contract_section=interface_contract_section,
            context_section=context_section,
        )
