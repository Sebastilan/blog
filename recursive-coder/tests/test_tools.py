"""测试目标：ToolExecutor 正确分发工具调用 + task_done 信号。

tools.py 是 Agent 和 Executor 之间的桥梁。
"""

import asyncio
import tempfile

import pytest

from recursive_coder.executor import Executor
from recursive_coder.tools import ToolExecutor, TOOL_DEFINITIONS


@pytest.fixture
def tool_exec(tmp_path):
    executor = Executor(workspace_dir=str(tmp_path), timeout=5)
    return ToolExecutor(executor)


class TestToolDefinitions:
    """工具定义格式正确性——如果格式不对，API 调用会直接报错。"""

    def test_all_tools_have_required_fields(self):
        for td in TOOL_DEFINITIONS:
            assert td["type"] == "function"
            fn = td["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn

    def test_tool_names(self):
        names = {td["function"]["name"] for td in TOOL_DEFINITIONS}
        assert names == {"shell", "write_file", "read_file", "list_dir", "task_done"}


class TestToolDispatch:
    """ToolExecutor 正确派发到 Executor 的对应方法。"""

    def test_shell(self, tool_exec):
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("shell", {"command": "echo dispatch_test"})
        )
        assert record.success
        assert "dispatch_test" in record.result_summary

    def test_write_then_read(self, tool_exec):
        asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("write_file", {"path": "demo.txt", "content": "hello tool"})
        )
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("read_file", {"path": "demo.txt"})
        )
        assert record.success
        assert "hello tool" in record.result_summary

    def test_task_done(self, tool_exec):
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("task_done", {"success": True, "summary": "all good"})
        )
        assert record.success
        assert record.result_summary == "all good"

    def test_task_done_failure(self, tool_exec):
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("task_done", {"success": False, "summary": "failed"})
        )
        assert not record.success

    def test_unknown_tool(self, tool_exec):
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("nonexistent_tool", {})
        )
        assert not record.success
        assert "Unknown" in record.result_summary

    def test_blocked_shell(self, tool_exec):
        record = asyncio.get_event_loop().run_until_complete(
            tool_exec.execute("shell", {"command": "rm -rf /"})
        )
        assert not record.success
        assert "BLOCKED" in record.result_summary
