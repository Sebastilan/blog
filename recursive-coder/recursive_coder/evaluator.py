"""Evaluation: collect metrics, generate reports, compare runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .logger_setup import get_logger
from .models import TaskNode, TaskStatus, TaskTree

logger = get_logger("evaluator")


class Evaluator:
    """Compute evaluation metrics from a completed task tree + API stats."""

    def __init__(self, workspace_dir: str) -> None:
        self.workspace = Path(workspace_dir)
        self.timeline: list[dict] = []
        self.start_time = time.time()

    def record_event(self, event_type: str, task_id: str = "", detail: str = "") -> None:
        self.timeline.append({
            "timestamp": time.time(),
            "event": event_type,
            "task_id": task_id,
            "detail": detail,
        })

    def generate_report(
        self,
        tree: TaskTree,
        api_stats: dict,
        config: dict,
        backtrack_count: int = 0,
    ) -> dict:
        """Generate the full evaluation report."""
        nodes = list(tree.nodes.values())
        leaves = [n for n in nodes if not n.children]
        intermediates = [n for n in nodes if n.children]

        # Efficiency
        latencies = sorted(api_stats.get("latencies", []))
        p50 = latencies[len(latencies) // 2] if latencies else 0
        p90 = latencies[int(len(latencies) * 0.9)] if latencies else 0

        total_tokens = api_stats.get("total_input_tokens", 0) + api_stats.get("total_output_tokens", 0)
        # DeepSeek pricing: ~$0.27/M input, ~$1.10/M output (cache miss)
        est_cost = (
            api_stats.get("total_input_tokens", 0) * 0.27 / 1_000_000
            + api_stats.get("total_output_tokens", 0) * 1.10 / 1_000_000
        )

        # Quality
        passed_leaves = [n for n in leaves if n.status == TaskStatus.PASSED]
        first_pass = [n for n in leaves if n.status == TaskStatus.PASSED and n.attempts <= 1]
        total_agent_steps = sum(n.agent_steps for n in leaves)
        avg_agent_steps = total_agent_steps / len(leaves) if leaves else 0

        # Tool usage distribution
        tool_dist: dict[str, int] = {}
        for n in nodes:
            for tc in n.tool_call_log:
                name = tc.get("tool_name", "?")
                tool_dist[name] = tool_dist.get(name, 0) + 1

        # Process
        max_depth = max((n.depth for n in nodes), default=0)
        max_width = max(
            (len(n.children) for n in nodes if n.children), default=0,
        )

        # Pass rate by depth
        depth_stats: dict[int, dict] = {}
        for n in leaves:
            d = n.depth
            if d not in depth_stats:
                depth_stats[d] = {"total": 0, "passed": 0}
            depth_stats[d]["total"] += 1
            if n.status == TaskStatus.PASSED:
                depth_stats[d]["passed"] += 1

        pass_by_depth = {
            str(d): round(s["passed"] / s["total"], 2) if s["total"] else 0
            for d, s in sorted(depth_stats.items())
        }

        # Shell command stats (from tool call logs across all nodes)
        shell_calls = 0
        shell_success = 0
        blocked_count = 0
        for n in nodes:
            for tc in n.tool_call_log:
                if tc.get("tool_name") == "shell":
                    shell_calls += 1
                    if tc.get("success"):
                        shell_success += 1
                    if "BLOCKED" in tc.get("result_summary", ""):
                        blocked_count += 1

        # Parse failures (from api_calls directory)
        parse_failures = self._count_parse_failures()

        report = {
            "run_id": self.workspace.name,
            "task_description": tree.nodes[tree.root_id].description if tree.root_id else "",
            "model": config.get("default_model", "?"),
            "status": self._overall_status(tree),
            "total_duration_seconds": round(time.time() - self.start_time, 1),

            "efficiency": {
                "total_api_calls": api_stats.get("total_calls", 0),
                "total_input_tokens": api_stats.get("total_input_tokens", 0),
                "total_output_tokens": api_stats.get("total_output_tokens", 0),
                "estimated_cost_usd": round(est_cost, 4),
                "api_latency_p50_ms": p50,
                "api_latency_p90_ms": p90,
                "avg_agent_steps": round(avg_agent_steps, 1),
            },

            "quality": {
                "first_pass_rate": round(len(first_pass) / len(leaves), 2) if leaves else 0,
                "avg_retries": round(
                    sum(n.attempts for n in leaves) / len(leaves), 2
                ) if leaves else 0,
                "backtrack_count": backtrack_count,
                "final_pass_rate": round(len(passed_leaves) / len(leaves), 2) if leaves else 0,
                "integration_pass_rate": round(
                    sum(1 for n in intermediates if n.status == TaskStatus.PASSED)
                    / len(intermediates), 2
                ) if intermediates else 1.0,
                "tool_usage_distribution": tool_dist,
            },

            "process": {
                "tree_max_depth": max_depth,
                "tree_max_width": max_width,
                "total_leaf_tasks": len(leaves),
                "total_intermediate_tasks": len(intermediates),
                "parse_failures": parse_failures,
                "pass_rate_by_depth": pass_by_depth,
                "shell_command_success_rate": round(
                    shell_success / shell_calls, 2
                ) if shell_calls else 1.0,
                "security_blocks": blocked_count,
            },

            "timeline": self.timeline[-200:],  # cap
        }

        return report

    def print_summary(self, report: dict) -> str:
        """Human-readable summary for terminal output."""
        lines = [
            "",
            "=" * 60,
            f"  Evaluation Report: {report.get('run_id', '?')}",
            "=" * 60,
            f"  Task:   {report.get('task_description', '')[:60]}",
            f"  Model:  {report.get('model', '?')}",
            f"  Status: {report.get('status', '?')}",
            f"  Time:   {report.get('total_duration_seconds', 0)}s",
            "",
            "  -- Efficiency --",
            f"  API calls:      {report['efficiency']['total_api_calls']}",
            f"  Tokens (in/out): {report['efficiency']['total_input_tokens']} / {report['efficiency']['total_output_tokens']}",
            f"  Est. cost:      ${report['efficiency']['estimated_cost_usd']}",
            f"  Latency P50:    {report['efficiency']['api_latency_p50_ms']}ms",
            f"  Avg agent steps: {report['efficiency']['avg_agent_steps']}",
            "",
            "  -- Quality --",
            f"  First-pass rate: {report['quality']['first_pass_rate']}",
            f"  Final pass rate: {report['quality']['final_pass_rate']}",
            f"  Avg retries:     {report['quality']['avg_retries']}",
            f"  Backtracks:      {report['quality']['backtrack_count']}",
            "",
            "  -- Process --",
            f"  Tree depth: {report['process']['tree_max_depth']}  width: {report['process']['tree_max_width']}",
            f"  Leaf tasks: {report['process']['total_leaf_tasks']}",
            f"  Parse failures:  {report['process']['parse_failures']}",
            f"  Shell success:   {report['process']['shell_command_success_rate']}",
            "=" * 60,
            "",
        ]
        return "\n".join(lines)

    def _overall_status(self, tree: TaskTree) -> str:
        if not tree.root_id:
            return "empty"
        root = tree.get_node(tree.root_id)
        if not root:
            return "empty"
        if root.status == TaskStatus.PASSED:
            return "completed"
        leaves = [n for n in tree.nodes.values() if not n.children]
        passed = sum(1 for n in leaves if n.status == TaskStatus.PASSED)
        if passed > 0:
            return "partial"
        return "failed"

    def _count_parse_failures(self) -> int:
        calls_dir = self.workspace / "api_calls"
        if not calls_dir.exists():
            return 0
        count = 0
        for f in calls_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if data.get("error"):
                    count += 1
            except Exception:
                pass
        return count
