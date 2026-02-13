"""测试目标：Executor 的安全过滤 + 超时 + 路径逃逸防护。

这些是安全边界，必须严格保证。
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from recursive_coder.executor import Executor


@pytest.fixture
def workspace(tmp_path):
    return tmp_path


@pytest.fixture
def executor(workspace):
    return Executor(
        workspace_dir=str(workspace),
        timeout=5,
        output_truncate=200,
        blacklist=["rm -rf /", "sudo ", "reboot"],
    )


class TestSafetyFilter:
    """黑名单命令必须被拦截。"""

    def test_block_rm_rf(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("rm -rf /")
        )
        assert result.blocked
        assert "blacklist" in result.block_reason

    def test_block_sudo(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("sudo apt install something")
        )
        assert result.blocked

    def test_allow_normal_command(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("echo hello")
        )
        assert not result.blocked
        assert result.returncode == 0
        assert "hello" in result.stdout


class TestCommandExecution:
    """基本命令执行。"""

    def test_capture_stdout(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("echo test123")
        )
        assert "test123" in result.stdout
        assert result.returncode == 0

    def test_capture_stderr(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("python -c \"import sys; sys.stderr.write('err_msg')\"")
        )
        assert "err_msg" in result.stderr

    def test_nonzero_exit(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("python -c \"exit(42)\"")
        )
        assert result.returncode == 42

    def test_timeout(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("sleep 30", timeout=1)
        )
        assert result.timed_out

    def test_output_truncation(self, executor):
        """输出超过 truncate 上限时应截断。"""
        result = asyncio.get_event_loop().run_until_complete(
            executor.run("python -c \"print('x' * 500)\"")
        )
        assert "truncated" in result.stdout
        assert len(result.stdout) <= 300  # 200 + truncation message


class TestFileOperations:
    """文件读写 + 路径逃逸防护。"""

    def test_write_and_read(self, executor, workspace):
        asyncio.get_event_loop().run_until_complete(
            executor.write_file("test.py", "print('hello')")
        )
        assert (workspace / "test.py").exists()

        content = asyncio.get_event_loop().run_until_complete(
            executor.read_file("test.py")
        )
        assert "print('hello')" in content

    def test_write_nested_dir(self, executor, workspace):
        """写入不存在的嵌套目录——应自动创建。"""
        result = asyncio.get_event_loop().run_until_complete(
            executor.write_file("a/b/c/deep.txt", "deep content")
        )
        assert "OK" in result
        assert (workspace / "a" / "b" / "c" / "deep.txt").exists()

    def test_path_escape_read(self, executor):
        """尝试读 workspace 外的文件——必须拒绝。"""
        result = asyncio.get_event_loop().run_until_complete(
            executor.read_file("../../../etc/passwd")
        )
        assert "ERROR" in result

    def test_path_escape_write(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.write_file("../../escape.txt", "bad")
        )
        assert "ERROR" in result

    def test_list_dir(self, executor, workspace):
        (workspace / "file1.txt").write_text("a")
        (workspace / "subdir").mkdir()
        result = asyncio.get_event_loop().run_until_complete(
            executor.list_dir(".")
        )
        assert "file1.txt" in result
        assert "subdir" in result

    def test_read_nonexistent(self, executor):
        result = asyncio.get_event_loop().run_until_complete(
            executor.read_file("no_such_file.txt")
        )
        assert "ERROR" in result
