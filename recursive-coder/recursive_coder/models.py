"""Core data structures: TaskNode, TaskTree, and data pipeline types."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Verification & Data Pipeline
# ---------------------------------------------------------------------------

@dataclass
class Verification:
    """How to verify a leaf task — always against real pipeline data."""

    description: str
    command: str = ""               # shell command to run verification
    expected_output: str = ""       # expected stdout (or file content)
    compare_mode: str = "contains"  # "exact" | "contains" | "file_diff" | "returncode"


@dataclass
class DataPort:
    """A task's input/output in the data pipeline.

    Verification data is NOT invented by AI.  It flows from the root node
    (which carries real test data) down through every decomposition.  Each
    child inherits concrete inputs from its parent or from a sibling's output.
    """

    input_description: str = ""
    input_files: list[str] = field(default_factory=list)
    output_description: str = ""
    output_files: list[str] = field(default_factory=list)
    expected_output_file: str = ""
    upstream_task_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_description": self.input_description,
            "input_files": self.input_files,
            "output_description": self.output_description,
            "output_files": self.output_files,
            "expected_output_file": self.expected_output_file,
            "upstream_task_ids": self.upstream_task_ids,
        }

    @classmethod
    def from_dict(cls, d: dict) -> DataPort:
        return cls(
            input_description=d.get("input_description", ""),
            input_files=d.get("input_files", []),
            output_description=d.get("output_description", ""),
            output_files=d.get("output_files", []),
            expected_output_file=d.get("expected_output_file", ""),
            upstream_task_ids=d.get("upstream_task_ids", []),
        )


@dataclass
class ToolCallRecord:
    """Single tool invocation inside an Agent loop."""

    tool_name: str
    arguments: dict
    result_summary: str = ""
    success: bool = True
    duration_ms: int = 0


@dataclass
class AgentStep:
    """One turn in the Agent loop."""

    step_number: int
    assistant_message: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    timestamp: float = 0.0


# ---------------------------------------------------------------------------
# TaskNode
# ---------------------------------------------------------------------------

@dataclass
class TaskNode:
    """A single node in the task tree."""

    description: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0

    # Verification (leaf tasks)
    verification: Optional[Verification] = None

    # Data pipeline — the key to real-data-driven verification
    data_port: DataPort = field(default_factory=DataPort)

    # Context files the model should read for this task
    context_files: list[str] = field(default_factory=list)
    # Files this task has produced
    output_files: list[str] = field(default_factory=list)

    # Sibling-level dependencies (task ids that must PASS before this one)
    dependencies: list[str] = field(default_factory=list)

    # Retry / backtrack
    attempts: int = 0
    max_attempts: int = 3
    error_log: list[str] = field(default_factory=list)
    decomposition_reason: str = ""

    # Timing & cost
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    api_call_ids: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})

    # Agent tracking
    agent_steps: int = 0
    tool_call_log: list[dict] = field(default_factory=list)

    # Planning — produced by judge phase
    execution_plan: list[str] = field(default_factory=list)  # step-by-step plan for leaf tasks
    interface: dict = field(default_factory=dict)  # input/output contract for this task
    interface_contract: str = ""  # shared contract between sibling tasks (set by parent's judge)

    # Misc
    implementation_hint: str = ""
    verification_result: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "depth": self.depth,
            "verification": {
                "description": self.verification.description,
                "command": self.verification.command,
                "expected_output": self.verification.expected_output,
                "compare_mode": self.verification.compare_mode,
            } if self.verification else None,
            "data_port": self.data_port.to_dict(),
            "context_files": self.context_files,
            "output_files": self.output_files,
            "dependencies": self.dependencies,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "error_log": self.error_log,
            "decomposition_reason": self.decomposition_reason,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "api_call_ids": self.api_call_ids,
            "token_usage": self.token_usage,
            "agent_steps": self.agent_steps,
            "tool_call_log": self.tool_call_log,
            "execution_plan": self.execution_plan,
            "interface": self.interface,
            "interface_contract": self.interface_contract,
            "implementation_hint": self.implementation_hint,
            "verification_result": self.verification_result,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskNode:
        verification = None
        if data.get("verification"):
            v = data["verification"]
            verification = Verification(
                description=v["description"],
                command=v.get("command", ""),
                expected_output=v.get("expected_output", ""),
                compare_mode=v.get("compare_mode", "contains"),
            )
        dp = data.get("data_port", {})
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            depth=data.get("depth", 0),
            verification=verification,
            data_port=DataPort.from_dict(dp),
            context_files=data.get("context_files", []),
            output_files=data.get("output_files", []),
            dependencies=data.get("dependencies", []),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            error_log=data.get("error_log", []),
            decomposition_reason=data.get("decomposition_reason", ""),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            api_call_ids=data.get("api_call_ids", []),
            token_usage=data.get("token_usage", {"input": 0, "output": 0}),
            agent_steps=data.get("agent_steps", 0),
            tool_call_log=data.get("tool_call_log", []),
            execution_plan=data.get("execution_plan", []),
            interface=data.get("interface", {}),
            interface_contract=data.get("interface_contract", ""),
            implementation_hint=data.get("implementation_hint", ""),
            verification_result=data.get("verification_result", ""),
        )


# ---------------------------------------------------------------------------
# TaskTree
# ---------------------------------------------------------------------------

class TaskTree:
    """Manages the full task tree."""

    def __init__(self) -> None:
        self.nodes: dict[str, TaskNode] = {}
        self.root_id: Optional[str] = None

    def add_node(self, node: TaskNode) -> None:
        self.nodes[node.id] = node
        if node.parent_id is None:
            self.root_id = node.id
        elif node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)

    def get_node(self, node_id: str) -> Optional[TaskNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[TaskNode]:
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_parent(self, node_id: str) -> Optional[TaskNode]:
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return None
        return self.nodes.get(node.parent_id)

    def topological_order(self, node_ids: list[str]) -> list[str]:
        id_set = set(node_ids)
        in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for nid in node_ids:
            node = self.nodes[nid]
            for dep in node.dependencies:
                if dep in id_set:
                    adj[dep].append(nid)
                    in_degree[nid] += 1
        queue = [nid for nid in node_ids if in_degree[nid] == 0]
        result: list[str] = []
        while queue:
            cur = queue.pop(0)
            result.append(cur)
            for nb in adj[cur]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)
        result.extend(nid for nid in node_ids if nid not in result)
        return result

    def get_ready_tasks(self, node_ids: list[str]) -> list[str]:
        ready = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.status != TaskStatus.PENDING:
                continue
            if all(
                self.nodes.get(d) and self.nodes[d].status == TaskStatus.PASSED
                for d in node.dependencies
            ):
                ready.append(nid)
        return ready

    def all_children_passed(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        return all(
            self.nodes[c].status == TaskStatus.PASSED
            for c in node.children if c in self.nodes
        )

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskTree:
        tree = cls()
        tree.root_id = data.get("root_id")
        for nid, nd in data.get("nodes", {}).items():
            tree.nodes[nid] = TaskNode.from_dict(nd)
        return tree

    def print_tree(self, node_id: Optional[str] = None, indent: int = 0) -> str:
        if node_id is None:
            node_id = self.root_id
        if not node_id:
            return "(empty tree)"
        node = self.nodes.get(node_id)
        if not node:
            return ""
        icons = {
            TaskStatus.PENDING: "[ ]", TaskStatus.RUNNING: "[~]",
            TaskStatus.PASSED: "[+]", TaskStatus.FAILED: "[X]",
        }
        pfx = "  " * indent
        lines = [f"{pfx}{icons[node.status]} {node.id}: {node.description}"]
        for cid in node.children:
            lines.append(self.print_tree(cid, indent + 1))
        return "\n".join(lines)
