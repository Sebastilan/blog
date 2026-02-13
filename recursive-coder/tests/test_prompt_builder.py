"""测试目标：PromptBuilder 正确注入数据管道内容到模板中。

数据管道是 v3 的核心改动，prompt 中必须正确携带输入数据。
"""

import tempfile
from pathlib import Path

from recursive_coder.models import DataPort, TaskNode, Verification
from recursive_coder.prompt_builder import PromptBuilder


class TestJudgePrompt:
    """judge prompt 必须包含任务的输入数据。"""

    def test_with_data_files(self, tmp_path):
        """当 data_port 指定了 input_files 且文件存在，prompt 中应包含文件内容。"""
        # 准备工作区和数据文件
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "input.txt").write_text("1 2 3 4 5")

        task = TaskNode(
            description="求和",
            data_port=DataPort(
                input_files=["data/input.txt"],
                input_description="一组整数",
            ),
        )

        builder = PromptBuilder()
        prompt = builder.judge(task, str(tmp_path))

        assert "求和" in prompt
        assert "1 2 3 4 5" in prompt  # 文件内容被注入

    def test_without_data(self):
        """没有输入数据时，prompt 应提示需要准备测试数据。"""
        task = TaskNode(description="写个复杂算法")
        builder = PromptBuilder()
        prompt = builder.judge(task, "/nonexistent")
        assert "prepare test data" in prompt.lower() or "准备" in prompt

    def test_with_context_files(self, tmp_path):
        """context_files 的内容也应出现在 prompt 中。"""
        (tmp_path / "api.h").write_text("int solve(int n);")

        task = TaskNode(
            description="实现 solve 函数",
            context_files=["api.h"],
            data_port=DataPort(input_description="n=5"),
        )

        builder = PromptBuilder()
        prompt = builder.judge(task, str(tmp_path))
        assert "int solve(int n)" in prompt


class TestExecutePrompt:
    """execute prompt 必须包含验证标准。"""

    def test_has_verification(self):
        task = TaskNode(
            description="实现加法",
            verification=Verification(
                description="测试 2+3=5",
                command="python test.py",
                expected_output="5",
            ),
        )
        builder = PromptBuilder()
        prompt = builder.execute(task, "/tmp")
        assert "python test.py" in prompt
        assert "5" in prompt


class TestFixPrompt:
    """fix prompt 必须包含错误信息。"""

    def test_has_error_info(self):
        task = TaskNode(
            description="修复加法",
            verification=Verification(
                description="test", command="python test.py", expected_output="5",
            ),
        )
        builder = PromptBuilder()
        prompt = builder.fix(task, "NameError: name 'add' is not defined", "/tmp")
        assert "NameError" in prompt
        assert "add" in prompt
