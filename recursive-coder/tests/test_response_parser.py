"""测试目标：response_parser 对各种模型返回格式的鲁棒性。

这是最容易出问题的模块——模型返回的格式千变万化。
每个 case 模拟一种真实场景下模型可能返回的格式。
"""

from recursive_coder.response_parser import (
    parse_judge_response,
    parse_backtrack_response,
    _extract_json_block,
)


class TestExtractJsonBlock:
    """底层 JSON 提取，三种策略都要能工作。"""

    def test_json_tag(self):
        text = '一些分析...\n<json>\n{"key": "value"}\n</json>\n后续文字'
        assert _extract_json_block(text) == {"key": "value"}

    def test_fenced_block(self):
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json_block(text) == {"key": "value"}

    def test_bare_json(self):
        """模型没用任何标记，直接输出 JSON。"""
        text = '这是结果 {"can_verify": true} 就这样'
        result = _extract_json_block(text)
        assert result == {"can_verify": True}

    def test_no_json_at_all(self):
        """模型完全没返回 JSON——这是真实会发生的。"""
        text = "我认为这个任务需要拆分，但我不确定怎么拆。"
        assert _extract_json_block(text) is None

    def test_malformed_json_fallback(self):
        """<json> 标签里的 JSON 格式错误，但后面有正确的裸 JSON。"""
        text = '<json>\n{broken json\n</json>\n{"can_verify": false, "subtasks": []}'
        result = _extract_json_block(text)
        assert result is not None
        assert result.get("can_verify") is False

    def test_nested_braces(self):
        """JSON 中有嵌套对象。"""
        text = '{"a": {"b": {"c": 1}}, "d": 2}'
        result = _extract_json_block(text)
        assert result["a"]["b"]["c"] == 1


class TestJudgeParser:
    """judge 阶段解析——区分"可验证"和"需拆分"两条路径。"""

    def test_can_verify(self):
        text = """
分析完了，可以直接实现。
<json>
{
  "can_verify": true,
  "verification": {
    "description": "测试 add 函数",
    "expected_output": "5",
    "command": "python -c \\"print(2+3)\\"",
    "compare_mode": "exact"
  },
  "data_port": {
    "input_description": "两个整数",
    "output_files": ["add.py"]
  },
  "implementation_hint": "一行就行"
}
</json>
"""
        r = parse_judge_response(text)
        assert r.parse_success
        assert r.can_verify
        assert r.verification_command == 'python -c "print(2+3)"'
        assert r.compare_mode == "exact"
        assert r.data_port["output_files"] == ["add.py"]

    def test_needs_decompose(self):
        text = """
<json>
{
  "can_verify": false,
  "subtasks": [
    {
      "description": "实现 tokenizer",
      "data_port": {"input_files": ["data/expr.txt"], "output_files": ["tokens.json"]},
      "dependencies": []
    },
    {
      "description": "实现 parser",
      "data_port": {"input_files": ["tokens.json"]},
      "dependencies": [0]
    }
  ],
  "decomposition_reason": "计算器需要分词和解析两步"
}
</json>
"""
        r = parse_judge_response(text)
        assert r.parse_success
        assert not r.can_verify
        assert len(r.subtasks) == 2
        assert r.subtasks[1]["dependencies"] == [0]
        assert "计算器" in r.decomposition_reason

    def test_parse_failure_graceful(self):
        """模型返回了完全无关的文字。不应崩溃。"""
        r = parse_judge_response("抱歉，我不太理解这个任务。能再说明一下吗？")
        assert not r.parse_success
        assert not r.can_verify
        assert r.raw_response != ""


class TestBacktrackParser:
    """backtrack 阶段解析。"""

    def test_normal(self):
        text = """
<json>
{
  "subtasks": [
    {"description": "新方案子任务1", "dependencies": []},
    {"description": "新方案子任务2", "dependencies": [0]}
  ],
  "decomposition_reason": "上次拆得太细了",
  "changes_from_previous": "合并了原来的 A 和 B"
}
</json>
"""
        r = parse_backtrack_response(text)
        assert r.parse_success
        assert len(r.subtasks) == 2
        assert "太细" in r.decomposition_reason

    def test_empty_subtasks(self):
        """模型返回了空的 subtasks 列表。"""
        text = '<json>{"subtasks": [], "decomposition_reason": "放弃了"}</json>'
        r = parse_backtrack_response(text)
        assert r.parse_success
        assert r.subtasks == []
