"""Core data structures for the task tree."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from typing import Optional


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class Verification:
    """Verification case for a leaf task."""

    description: str
    input_data: str = ""
    expected_output: str = ""
    command: str = ""


@dataclass
class TaskNode:
    """A single node in the task tree."""

    description: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: TaskStatus = TaskStatus.PENDING
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    verification: Optional[Verification] = None
    context_files: list[str] = field(default_factory=list)
    output_files: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    depth: int = 0
    dependencies: list[str] = field(default_factory=list)
    error_log: list[str] = field(default_factory=list)
    implementation_hint: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "children": self.children,
            "verification": {
                "description": self.verification.description,
                "input_data": self.verification.input_data,
                "expected_output": self.verification.expected_output,
                "command": self.verification.command,
            }
            if self.verification
            else None,
            "context_files": self.context_files,
            "output_files": self.output_files,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "depth": self.depth,
            "dependencies": self.dependencies,
            "error_log": self.error_log,
            "implementation_hint": self.implementation_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskNode:
        verification = None
        if data.get("verification"):
            v = data["verification"]
            verification = Verification(
                description=v["description"],
                input_data=v.get("input_data", ""),
                expected_output=v.get("expected_output", ""),
                command=v.get("command", ""),
            )
        return cls(
            id=data["id"],
            description=data["description"],
            status=TaskStatus(data["status"]),
            parent_id=data.get("parent_id"),
            children=data.get("children", []),
            verification=verification,
            context_files=data.get("context_files", []),
            output_files=data.get("output_files", []),
            attempts=data.get("attempts", 0),
            max_attempts=data.get("max_attempts", 3),
            depth=data.get("depth", 0),
            dependencies=data.get("dependencies", []),
            error_log=data.get("error_log", []),
            implementation_hint=data.get("implementation_hint", ""),
        )


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
        """Return node_ids in dependency-respecting order (simple topological sort)."""
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
            current = queue.pop(0)
            result.append(current)
            for neighbor in adj[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If there are remaining nodes (cycle), append them anyway
        remaining = [nid for nid in node_ids if nid not in result]
        result.extend(remaining)
        return result

    def get_ready_tasks(self, node_ids: list[str]) -> list[str]:
        """Return tasks whose dependencies are all passed."""
        ready = []
        for nid in node_ids:
            node = self.nodes[nid]
            if node.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                self.nodes.get(dep) and self.nodes[dep].status == TaskStatus.PASSED
                for dep in node.dependencies
            )
            if deps_met:
                ready.append(nid)
        return ready

    def all_children_passed(self, node_id: str) -> bool:
        node = self.nodes.get(node_id)
        if not node:
            return False
        return all(
            self.nodes[cid].status == TaskStatus.PASSED
            for cid in node.children
            if cid in self.nodes
        )

    def to_dict(self) -> dict:
        return {
            "root_id": self.root_id,
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> TaskTree:
        tree = cls()
        tree.root_id = data.get("root_id")
        for nid, node_data in data.get("nodes", {}).items():
            tree.nodes[nid] = TaskNode.from_dict(node_data)
        return tree

    def print_tree(self, node_id: Optional[str] = None, indent: int = 0) -> str:
        """Return a text visualization of the task tree."""
        if node_id is None:
            node_id = self.root_id
        if node_id is None:
            return "(empty tree)"

        node = self.nodes.get(node_id)
        if not node:
            return ""

        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.RUNNING: "[~]",
            TaskStatus.PASSED: "[+]",
            TaskStatus.FAILED: "[X]",
        }

        prefix = "  " * indent
        lines = [f"{prefix}{status_icon[node.status]} {node.id}: {node.description}"]
        for cid in node.children:
            lines.append(self.print_tree(cid, indent + 1))
        return "\n".join(lines)
