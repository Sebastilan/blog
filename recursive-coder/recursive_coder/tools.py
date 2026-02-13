"""Tool definitions (OpenAI function-calling format) and tool dispatch."""

from __future__ import annotations

import json
import time
from typing import Any

from .executor import Executor
from .logger_setup import get_logger
from .models import ToolCallRecord

logger = get_logger("tools")

# ── OpenAI-compatible tool definitions ──────────────────────────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Execute a shell command in the working directory. Use for compiling, running tests, installing packages, inspecting files, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory",
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to the working directory (default: '.')",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_done",
            "description": "Declare that the current task is finished. Call this when the verification passes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was done",
                    },
                },
                "required": ["success", "summary"],
            },
        },
    },
]


class ToolExecutor:
    """Dispatch tool calls to the underlying Executor."""

    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    async def execute(self, tool_name: str, arguments: dict) -> ToolCallRecord:
        t0 = time.monotonic()
        try:
            if tool_name == "shell":
                result = await self.executor.run(arguments["command"])
                summary = result.stdout or result.stderr or "(no output)"
                if result.blocked:
                    summary = f"BLOCKED: {result.block_reason}"
                success = result.returncode == 0 and not result.blocked
            elif tool_name == "write_file":
                summary = await self.executor.write_file(
                    arguments["path"], arguments["content"],
                )
                success = summary.startswith("OK")
            elif tool_name == "read_file":
                summary = await self.executor.read_file(arguments["path"])
                success = not summary.startswith("ERROR")
            elif tool_name == "list_dir":
                summary = await self.executor.list_dir(arguments.get("path", "."))
                success = not summary.startswith("ERROR")
            elif tool_name == "task_done":
                summary = arguments.get("summary", "")
                success = arguments.get("success", False)
            else:
                summary = f"Unknown tool: {tool_name}"
                success = False
        except Exception as exc:
            summary = f"Tool execution error: {exc}"
            success = False

        duration = int((time.monotonic() - t0) * 1000)
        logger.debug("tool=%s success=%s duration=%dms", tool_name, success, duration)
        return ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result_summary=summary,
            success=success,
            duration_ms=duration,
        )
