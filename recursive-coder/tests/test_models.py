"""测试目标：DataPort 数据管道 + TaskTree 拓扑排序 + 序列化完整性。

这三个是框架的数据基础，任何一个坏了整个流程都跑不通。
"""

import json
from recursive_coder.models import (
    DataPort, TaskNode, TaskStatus, TaskTree, Verification,
)


class TestDataPortRoundTrip:
    """DataPort 是 v3 新增的核心结构，必须确保序列化不丢字段。"""

    def test_full_fields(self):
        dp = DataPort(
            input_description="CVRP 实例文件",
            input_files=["data/vrp5.txt", "data/config.json"],
            output_description="距离矩阵 + 需求向量",
            output_files=["output/dist_matrix.csv"],
            expected_output_file="data/expected_dist.csv",
            upstream_task_ids=["abc123"],
        )
        d = dp.to_dict()
        dp2 = DataPort.from_dict(d)
        assert dp2.input_files == ["data/vrp5.txt", "data/config.json"]
        assert dp2.upstream_task_ids == ["abc123"]
        assert dp2.expected_output_file == "data/expected_dist.csv"

    def test_empty_defaults(self):
        """from_dict 传空 dict 不应崩溃。"""
        dp = DataPort.from_dict({})
        assert dp.input_files == []
        assert dp.output_description == ""


class TestTaskNodeSerialization:
    """TaskNode 字段很多，重点验证 verification + data_port 的嵌套序列化。"""

    def test_full_round_trip(self):
        node = TaskNode(
            description="解析输入数据",
            verification=Verification(
                description="检查解析结果",
                command="python check.py",
                expected_output="OK",
                compare_mode="exact",
            ),
            data_port=DataPort(
                input_files=["data/in.txt"],
                output_files=["output/parsed.json"],
            ),
            attempts=2,
            error_log=["error1", "error2"],
            agent_steps=5,
            token_usage={"input": 1000, "output": 500},
        )
        d = node.to_dict()
        # 关键：确保 JSON 可序列化（不能有 dataclass 对象残留）
        json_str = json.dumps(d)
        restored = TaskNode.from_dict(json.loads(json_str))

        assert restored.verification.compare_mode == "exact"
        assert restored.data_port.output_files == ["output/parsed.json"]
        assert restored.token_usage == {"input": 1000, "output": 500}
        assert restored.error_log == ["error1", "error2"]

    def test_none_verification(self):
        """verification 为 None 时序列化不应崩溃。"""
        node = TaskNode(description="no verification")
        d = node.to_dict()
        assert d["verification"] is None
        restored = TaskNode.from_dict(d)
        assert restored.verification is None


class TestTopologicalSort:
    """拓扑排序是并行执行和依赖管理的基础。
    重点测试：有依赖、有环、空列表。
    """

    def _build_chain(self):
        """A → B → C 的依赖链"""
        tree = TaskTree()
        root = TaskNode(description="root", id="root")
        tree.add_node(root)
        a = TaskNode(description="A", id="a", parent_id="root")
        b = TaskNode(description="B", id="b", parent_id="root", dependencies=["a"])
        c = TaskNode(description="C", id="c", parent_id="root", dependencies=["b"])
        for n in [a, b, c]:
            tree.add_node(n)
        return tree

    def test_linear_chain(self):
        tree = self._build_chain()
        order = tree.topological_order(["a", "b", "c"])
        assert order.index("a") < order.index("b") < order.index("c")

    def test_parallel_tasks(self):
        """A 和 B 无依赖，C 依赖两者。"""
        tree = TaskTree()
        root = TaskNode(description="root", id="root")
        tree.add_node(root)
        a = TaskNode(description="A", id="a", parent_id="root")
        b = TaskNode(description="B", id="b", parent_id="root")
        c = TaskNode(description="C", id="c", parent_id="root", dependencies=["a", "b"])
        for n in [a, b, c]:
            tree.add_node(n)
        order = tree.topological_order(["a", "b", "c"])
        assert order.index("c") > order.index("a")
        assert order.index("c") > order.index("b")

    def test_cycle_does_not_hang(self):
        """循环依赖不应死循环，应 fallback 输出所有节点。"""
        tree = TaskTree()
        root = TaskNode(description="root", id="root")
        tree.add_node(root)
        a = TaskNode(description="A", id="a", parent_id="root", dependencies=["b"])
        b = TaskNode(description="B", id="b", parent_id="root", dependencies=["a"])
        tree.add_node(a)
        tree.add_node(b)
        order = tree.topological_order(["a", "b"])
        assert set(order) == {"a", "b"}  # 都在，不丢

    def test_empty_list(self):
        tree = TaskTree()
        assert tree.topological_order([]) == []


class TestGetReadyTasks:
    """get_ready_tasks 是并行调度的核心判据。"""

    def test_deps_not_met(self):
        tree = TaskTree()
        root = TaskNode(description="root", id="root")
        a = TaskNode(description="A", id="a", parent_id="root", status=TaskStatus.PENDING)
        b = TaskNode(description="B", id="b", parent_id="root",
                     status=TaskStatus.PENDING, dependencies=["a"])
        for n in [root, a, b]:
            tree.add_node(n)
        ready = tree.get_ready_tasks(["a", "b"])
        assert ready == ["a"]  # b 的依赖 a 还没 PASSED

    def test_deps_met(self):
        tree = TaskTree()
        root = TaskNode(description="root", id="root")
        a = TaskNode(description="A", id="a", parent_id="root", status=TaskStatus.PASSED)
        b = TaskNode(description="B", id="b", parent_id="root",
                     status=TaskStatus.PENDING, dependencies=["a"])
        for n in [root, a, b]:
            tree.add_node(n)
        ready = tree.get_ready_tasks(["a", "b"])
        assert ready == ["b"]  # a 已 PASSED，b 可执行
