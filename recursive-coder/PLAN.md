# 递归任务分解 AI 编码框架 — 实施计划 v3

> **与 v2 的核心差异：**
> 1. 执行模式从"只写代码"改为 **Agent 模式**——模型可以自由调用终端工具
> 2. 新增 **迭代优化闭环**——评估报告自动驱动 prompt/参数调整
> 3. ★★★ **数据管道驱动验证**——验证数据不是 AI 编造的，而是从根节点真实数据逐层流下来的

---

## 零、核心设计变更：数据管道驱动验证

### 问题

v2 的验证模式是让 AI 为每个子任务"编造"验证用例。这有根本缺陷：
- 模型可能编造一个极简 case 来骗过验证，实际代码处理不了真实数据
- 子任务之间的数据接口对不上（A 的输出格式和 B 的预期输入不一致）
- 集成时才发现问题，返工成本高

### 解决：数据管道

**真实测试数据从根节点进入，逐层分解为子任务的输入/输出。**

```
根任务：求解 CVRP
├── 输入数据：vrp_instance.txt（真实的 5 客户小实例）
├── 预期输出：最优路线 + 总成本
│
├── 子任务 1：解析输入
│   ├── 输入：vrp_instance.txt（继承自根）
│   └── 输出：distance_matrix, demands, capacity（具体数值已知）
│
├── 子任务 2：构建初始 LP
│   ├── 输入：子任务 1 的输出（具体数值）
│   └── 输出：LP 模型文件 / 初始解
│
├── 子任务 3：定价子问题 (ESPPRC)
│   ├── 输入：对偶值（从 LP 解算出，具体数值已知）+ distance_matrix
│   └── 输出：负 reduced cost 的路径列表
│
└── 子任务 4：主循环 + 分支定界
    ├── 输入：所有上游模块
    └── 输出：最终解（可与根节点的预期输出比对）
```

### 关键规则

1. **根节点必须有真实测试数据**
   - 用户提供：最佳情况
   - 用户不提供：**"准备测试数据"自动成为第一个子任务**
   - "准备测试数据" 这个子任务由框架辅助完成（搜索公开数据集、生成合理的小规模测试实例）

2. **拆分时必须定义数据流**
   - judge prompt 不只是问"能不能验证"
   - 而是问"给定这份输入数据，任务怎么拆，每个子任务的输入/输出是什么"
   - 每个子任务的输入要么来自父节点的输入数据，要么来自兄弟节点的输出

3. **验证 = 跑真实数据 + 对比预期输出**
   - 叶子任务的验证：拿真实输入跑，输出和预期比对
   - 中间节点的集成验证：拿根节点的完整数据端到端跑

4. **数据文件存在文件系统中**
   - 输入数据写入 `workspace/<run_id>/data/`
   - 每个子任务的预期输出也写入此目录
   - 子任务可以通过 `context_files` 引用这些数据文件

### TaskNode 中的数据流字段

```python
# 数据管道
input_data_files: list[str] = []     # 该任务的输入数据文件路径
expected_output_file: str = ""        # 该任务的预期输出文件路径
actual_output_file: str = ""          # 实际运行后的输出文件路径
upstream_tasks: list[str] = []        # 输入来自哪些任务的输出
```

### judge prompt 的变化

旧的问法：
> "你能否为这个任务构造验证用例？"

新的问法：
> "这是任务描述，这是输入数据（具体值）。
>  你能否直接实现这个任务？如果能，预期输出是什么？
>  如果不能，怎么拆分？每个子任务的输入从哪来、输出是什么？"

---

## 一、架构总览

### 1.1 两层循环

```
外层循环（迭代优化）
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Run N                                             │
│   ┌───────────────────────────────────────┐         │
│   │  内层循环（任务递归）                     │         │
│   │                                       │         │
│   │  用户任务 → 拆分 → 执行(Agent) → 验证   │         │
│   │       ↑                    ↓          │         │
│   │       └──── 回溯 ←── 失败 ──┘          │         │
│   └───────────────────────────────────────┘         │
│                     │                               │
│                     ▼                               │
│              evaluation_report.json                  │
│                     │                               │
│                     ▼                               │
│              optimizer.py（分析报告，生成调整建议）       │
│                     │                               │
│                     ▼                               │
│              更新 prompt_templates / config           │
│                     │                               │
│                     ▼                               │
│   Run N+1 （使用优化后的配置）                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.2 Agent 模式 vs 旧方案对比

| 维度 | v1（只写代码） | v2（Agent 模式） |
|------|--------------|----------------|
| 执行方式 | 模型一次性输出完整代码 | 模型在多轮 tool-use 循环中逐步操作 |
| 终端交互 | 仅系统运行验证命令 | 模型可请求执行任意 shell 命令 |
| 文件操作 | 模型输出代码，系统写入 | 模型可主动读/写/查看文件 |
| 环境感知 | 无 | 模型可 `ls`、`cat`、`pip list` 等探索环境 |
| 调试能力 | 仅看到验证失败的 stderr | 模型可自主运行调试命令、检查中间输出 |
| 上下文 | 单次调用 | 单任务内多轮对话，但任务间独立 |

### 1.3 项目文件结构

```
recursive-coder/
├── pyproject.toml
├── config.yaml                        # 运行配置
├── PLAN.md
│
├── recursive_coder/
│   ├── __init__.py
│   ├── models.py                      # 数据结构（TaskNode, TaskTree, ToolCall, AgentStep）
│   ├── api_caller.py                  # LLM API 统一调用层（支持 tool_use 格式）
│   ├── tools.py                       # ★ Tool 定义与执行（shell, read_file, write_file, list_dir）
│   ├── agent_loop.py                  # ★ Agent 多轮循环（调 API → 解析 tool_call → 执行 → 回传结果）
│   ├── prompt_builder.py              # Prompt 构造（judge/execute/fix/backtrack/integrate）
│   ├── response_parser.py             # 解析判断/拆分阶段的结构化返回
│   ├── processor.py                   # 核心递归处理器（process/execute/backtrack）
│   ├── executor.py                    # Shell 安全执行层（超时、输出截断、命令过滤）
│   ├── persistence.py                 # 任务树 + API 调用记录持久化
│   ├── evaluator.py                   # ★ 评估器：收集指标、生成评估报告
│   ├── optimizer.py                   # ★★ 迭代优化器：分析报告 → 生成调整建议 → 更新配置
│   ├── logger_setup.py                # 结构化日志
│   ├── cli.py                         # CLI 入口
│   └── __main__.py                    # python -m recursive_coder
│
├── prompt_templates/                   # ★ Prompt 模板（独立文件，方便迭代修改）
│   ├── system.txt                     # 系统模板
│   ├── judge.txt                      # 判断/拆分 prompt
│   ├── execute.txt                    # 执行 prompt（Agent 模式指令）
│   ├── fix.txt                        # 修复 prompt
│   ├── backtrack.txt                  # 回溯 prompt
│   └── integrate.txt                  # 集成验证 prompt
│
├── optimization_history/               # ★ 优化历史记录
│   ├── iteration_001.json             # 第一轮优化记录
│   ├── iteration_002.json
│   └── ...
│
├── workspace/                          # 运行时工作区
│   └── <run_id>/
│       ├── task_tree.json
│       ├── run.log
│       ├── evaluation_report.json
│       ├── api_calls/
│       │   └── call_XXX.json
│       └── output/                    # 生成的代码产物
│
└── tests/
    ├── test_models.py
    ├── test_tools.py
    ├── test_agent_loop.py
    ├── test_response_parser.py
    └── test_executor.py
```

---

## 二、Agent 模式详细设计

### 2.1 Tool 定义

模型可使用的工具集：

| Tool 名称 | 参数 | 说明 | 安全级别 |
|-----------|------|------|---------|
| `shell` | `command: str` | 执行 shell 命令 | 需过滤 |
| `write_file` | `path: str, content: str` | 写文件 | 限制在工作区 |
| `read_file` | `path: str` | 读文件 | 限制在工作区 |
| `list_dir` | `path: str` | 列出目录内容 | 限制在工作区 |
| `task_done` | `success: bool, summary: str` | 声明任务完成 | 无限制 |

**shell 命令安全过滤规则：**

```
允许（白名单模式）:
  - 编译: gcc, g++, make, cmake
  - Python: python, pip install, pytest
  - 文件查看: cat, head, tail, wc, diff
  - 搜索: grep, find, ls
  - 其他: cd, echo, mkdir, cp, mv

禁止（黑名单）:
  - 危险操作: rm -rf, sudo, chmod 777
  - 网络请求: curl, wget, nc, ssh（防止数据泄露）
  - 系统修改: systemctl, kill, reboot

可配置:
  - 用户可在 config.yaml 中自定义白名单/黑名单
  - 默认 "严格模式"（白名单），可切换为 "宽松模式"（仅黑名单）
```

### 2.2 Agent 循环（单任务执行）

```
agent_loop(task, system_prompt, tools):

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt},
    ]

    for step in range(max_agent_steps):
        response = call_api(messages, tools=tool_definitions)

        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "tool", "content": result})
                evaluator.record_tool_call(tool_call, result)

        elif response.is_task_done:
            return response.summary

        else:
            # 模型纯文本回复，视为思考过程，继续循环
            messages.append({"role": "assistant", "content": response.text})
            messages.append({"role": "user", "content": "请继续执行，使用工具完成任务。"})

    # 超过最大步数，标记失败
    return TaskFailed("exceeded max agent steps")
```

**关键设计：Agent 循环的上下文管理**

问题：多轮对话会导致上下文不断增长，最终超出窗口限制。

解决：**滑动窗口 + 摘要压缩**
- 保留最近 N 轮完整对话（默认 N=10）
- 超出的历史轮次压缩为摘要（"之前你已经完成了：安装依赖、创建文件结构..."）
- 系统 prompt 和当前任务描述始终保留

```python
def compress_messages(messages, keep_recent=10):
    if len(messages) <= keep_recent * 2 + 2:  # system + user + recent
        return messages

    system = messages[0]
    task = messages[1]
    old = messages[2:-keep_recent*2]
    recent = messages[-keep_recent*2:]

    summary = summarize(old)  # 用模型或规则生成摘要
    return [system, task, {"role": "user", "content": f"[历史摘要] {summary}"}] + recent
```

### 2.3 API 调用格式（以 DeepSeek 为例）

DeepSeek 兼容 OpenAI 格式的 function calling：

```json
{
  "model": "deepseek-chat",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "shell",
        "description": "在工作目录中执行 shell 命令",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string", "description": "要执行的命令"}
          },
          "required": ["command"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "write_file",
        "description": "写入文件到工作目录",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string", "description": "相对于工作目录的文件路径"},
            "content": {"type": "string", "description": "文件内容"}
          },
          "required": ["path", "content"]
        }
      }
    }
  ]
}
```

**注意：** 判断/拆分阶段 **不提供** tools（不需要 Agent 能力），只有执行阶段才启用。

---

## 三、迭代优化闭环详细设计 ★★

这是本框架的独特价值所在。大多数 AI 编码工具只做单次运行，没有系统化的自我改进机制。

### 3.1 迭代优化流程

```
┌──────────────────────────────────────────────────────┐
│ 第 1 轮：用默认配置跑测试任务                            │
│     ↓                                                │
│ 生成 evaluation_report.json                           │
│     ↓                                                │
│ optimizer.py 分析报告，识别问题                          │
│     ↓                                                │
│ 生成 optimization_suggestion.json                     │
│     ↓                                                │
│ 自动/人工审核 → 更新 prompt_templates + config.yaml     │
│     ↓                                                │
│ 第 2 轮：用优化后配置跑 **同一个** 测试任务               │
│     ↓                                                │
│ 对比 report_v1 vs report_v2 → 指标是否改善？            │
│     ↓                                                │
│ 继续迭代 or 切换到更难的测试任务                          │
└──────────────────────────────────────────────────────┘
```

### 3.2 optimizer.py 的工作方式

optimizer 本身也调用 LLM（可以是同一个模型，也可以是更强的模型），输入是：

```
输入:
  1. 本轮 evaluation_report.json
  2. 当前的 prompt_templates/*
  3. 当前的 config.yaml
  4. （如果非首轮）上一轮的 report + 上一轮的调整记录

输出:
  optimization_suggestion.json:
  {
    "analysis": {
      "main_issues": [
        "一次通过率只有 55%，主要失败在第 3 层叶子任务",
        "parse_failures 达到 12 次，模型经常不按格式返回",
        "平均 agent 步数 15，很多步骤是重复的无效操作"
      ],
      "root_causes": [
        "execute.txt 的 prompt 没有明确要求模型先看目录结构再写代码",
        "judge.txt 的 JSON 格式示例不够清晰",
        "没有限制模型在单个任务中的最大文件数"
      ]
    },
    "prompt_changes": [
      {
        "file": "execute.txt",
        "change_type": "append",
        "content": "在开始编码前，请先用 list_dir 和 read_file 了解现有文件结构。",
        "reason": "减少模型盲写代码导致的路径错误"
      },
      {
        "file": "judge.txt",
        "change_type": "replace",
        "old": "请将你的判断结果包裹在...",
        "new": "你必须严格按以下 JSON 格式返回...(更详细的示例)",
        "reason": "降低 parse_failures"
      }
    ],
    "config_changes": [
      {
        "key": "max_agent_steps",
        "old": 30,
        "new": 20,
        "reason": "强制模型更高效地行动，减少无效步骤"
      }
    ],
    "expected_improvements": {
      "first_pass_rate": "55% → 70%",
      "parse_failures": "12 → 3",
      "avg_agent_steps": "15 → 10"
    }
  }
```

### 3.3 优化历史追踪

每轮优化的完整记录保存在 `optimization_history/iteration_NNN.json`：

```json
{
  "iteration": 1,
  "timestamp": "2026-02-13T15:00:00Z",
  "test_task": "用 Python 实现一个简单的计算器",
  "model": "deepseek-v3",

  "before": {
    "run_id": "20260213_140000",
    "first_pass_rate": 0.55,
    "parse_failures": 12,
    "total_cost_usd": 0.05,
    "final_pass_rate": 0.80
  },

  "changes_applied": {
    "prompt_changes": [...],
    "config_changes": [...]
  },

  "after": {
    "run_id": "20260213_150000",
    "first_pass_rate": 0.72,
    "parse_failures": 3,
    "total_cost_usd": 0.04,
    "final_pass_rate": 0.95
  },

  "improvement": {
    "first_pass_rate": "+0.17",
    "parse_failures": "-9",
    "total_cost_usd": "-0.01",
    "final_pass_rate": "+0.15"
  },

  "verdict": "improved"
}
```

### 3.4 迭代模式 CLI

```bash
# 单次运行
python -m recursive_coder run "实现一个计算器"

# 迭代优化模式（自动运行 N 轮，每轮之间自动优化）
python -m recursive_coder iterate "实现一个计算器" --rounds 3

# 查看优化历史
python -m recursive_coder history

# 对比两次运行
python -m recursive_coder compare run_20260213_140000 run_20260213_150000

# 用上一轮优化后的配置，换一个更难任务
python -m recursive_coder run "实现一个 TODO 应用" --inherit-config
```

### 3.5 自动 vs 人工审核

默认模式：**半自动**
- optimizer 生成建议后，打印摘要到终端
- 等待用户确认（`--auto-apply` 可跳过确认，全自动）
- 用户可以手动修改建议后再应用

全自动模式（`iterate --auto`）：
- optimizer 的建议直接应用
- 适合无人值守的批量实验

---

## 四、评估指标体系（完善版）

在 v1 基础上新增 Agent 模式专属指标。

### 4.1 效率指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 总 API 调用次数 | 直接计数 | 整体开销 |
| 总 token 消耗 | input + output | 成本 |
| 预估费用 | token × 单价 | 直观成本 |
| 总耗时 | wall clock | 端到端时间 |
| API 延迟 P50/P90 | 统计 | 识别慢调用 |
| **平均 Agent 步数** | ★ 每个叶子任务的平均 tool-use 轮数 | Agent 效率 |
| **无效步数比** | ★ 未产生有效文件变更的步数 / 总步数 | Agent 浪费程度 |
| 有效调用比 | 成功调用 / 总调用 | 是否浪费 |

### 4.2 质量指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 一次通过率 | 首次验证通过 / 总叶子 | 代码质量 |
| 平均重试次数 | attempts 均值 | 重试代价 |
| 回溯次数 | backtrack 触发次数 | 拆分质量 |
| 最终通过率 | PASSED / 总叶子 | 完成度 |
| 集成验证通过率 | 中间节点通过率 | 组合质量 |
| **Agent 自修复率** | ★ Agent 在循环内自行发现并修复错误的次数 | 自主调试能力 |
| **工具使用分布** | ★ 各 tool 的调用频率 | 理解 Agent 行为模式 |

### 4.3 过程指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 任务树深度 | max depth | 拆分深度 |
| 任务树宽度 | max children | 拆分广度 |
| 叶子任务数 | leaf count | 粒度 |
| 解析失败次数 | parse error count | prompt 质量 |
| 按深度通过率 | 各层 pass/fail | 薄弱环节 |
| **shell 命令成功率** | ★ 命令执行成功 / 总命令 | 模型对环境的理解程度 |
| **安全拦截次数** | ★ 被过滤的危险命令数 | 安全边界触发频率 |

### 4.4 迭代指标（跨运行对比）

| 指标 | 意义 |
|------|------|
| 迭代改善率 | 连续 N 轮中指标的趋势 |
| 收敛速度 | 几轮后指标趋于稳定 |
| prompt 变更次数 | 哪些 prompt 被改动最多（即最不稳定） |
| 最大瓶颈 | 连续多轮仍未解决的问题 |

---

## 五、各模块实施细节

### Step 1: pyproject.toml + config.yaml + .gitignore

**依赖：**
- `httpx` — 异步 HTTP
- `pyyaml` — 配置解析
- 标准库：asyncio, json, logging, subprocess, argparse, dataclasses, uuid, time, pathlib, re

**config.yaml：**
```yaml
# 模型配置
default_model: "deepseek-v3"
judge_model: null              # null 则用 default_model
execute_model: null
optimizer_model: null

# 递归控制
max_depth: 5
max_retries: 3
max_backtrack_retries: 2
max_total_api_calls: 500

# Agent 控制
max_agent_steps: 30            # 单任务最大 tool-use 轮数
agent_context_window: 10       # 保留最近 N 轮完整对话
command_timeout: 60            # shell 命令超时秒数
output_truncate: 10000         # stdout/stderr 截断字符数

# 安全
security_mode: "strict"        # strict(白名单) | permissive(黑名单)
command_whitelist: []           # 额外允许的命令（追加到默认白名单）
command_blacklist: []           # 额外禁止的命令（追加到默认黑名单）

# 迭代优化
auto_apply_optimization: false  # 是否自动应用优化建议
optimization_rounds: 3          # iterate 命令的默认轮数

# 并行
parallel: false
max_parallel_tasks: 3
```

### Step 2: logger_setup.py

- 控制台：简洁，`[HH:MM:SS] [LEVEL] [task_id] message`
- 文件：完整，写入 `workspace/<run_id>/run.log`
- DEBUG 级别记录完整 prompt/response（仅文件）

### Step 3: models.py 完善

新增字段（在现有 TaskNode 基础上）：

```python
# 时间追踪
start_time: Optional[float] = None
end_time: Optional[float] = None

# API 调用追踪
api_call_ids: list[str] = field(default_factory=list)
token_usage: dict = field(default_factory=lambda: {"input": 0, "output": 0})

# Agent 模式追踪
agent_steps: int = 0                # 该任务的 tool-use 轮数
tool_calls: list[dict] = field(default_factory=list)  # [{tool, args, result_summary}]

# 分解追踪
decomposition_reason: str = ""
verification_result: str = ""
```

新增数据类：

```python
@dataclass
class ToolCall:
    """单次工具调用记录"""
    tool_name: str
    arguments: dict
    result: str
    success: bool
    duration_ms: int

@dataclass
class AgentStep:
    """Agent 循环中的一步"""
    step_number: int
    assistant_message: str
    tool_calls: list[ToolCall]
    timestamp: float
```

### Step 4: api_caller.py 完善

关键变更：
- 支持 **tool_use / function_calling** 格式
- `call()` 方法新增参数：`tools`, `task_node_id`, `phase`
- 每次调用记录写入 `api_calls/call_XXX.json`
- DeepSeek 作为默认模型

```python
async def call(
    self,
    messages: list[dict],           # 改为接受完整 messages（支持多轮）
    tools: Optional[list[dict]] = None,  # tool definitions
    model_name: Optional[str] = None,
    task_node_id: str = "",
    phase: str = "",
) -> dict:
    """返回完整的 response dict（包含 tool_calls 或 text）"""
```

### Step 5: tools.py — Tool 定义与执行

```python
TOOL_DEFINITIONS = [...]  # OpenAI function calling 格式的 tool 定义

class ToolExecutor:
    def __init__(self, workspace_dir: str, config: dict)

    async def execute(self, tool_name: str, arguments: dict) -> ToolCall:
        """执行一个 tool call，返回结果"""

    def _check_command_safety(self, command: str) -> tuple[bool, str]:
        """检查命令安全性，返回 (allowed, reason)"""

    async def _shell(self, command: str) -> str
    async def _write_file(self, path: str, content: str) -> str
    async def _read_file(self, path: str) -> str
    async def _list_dir(self, path: str) -> str
```

### Step 6: agent_loop.py — Agent 多轮循环

```python
class AgentLoop:
    def __init__(self, api_caller, tool_executor, evaluator, config)

    async def run(self, task: TaskNode, system_prompt: str, user_prompt: str) -> AgentResult:
        """运行 Agent 循环直到模型调用 task_done 或超过步数限制"""

    def _compress_messages(self, messages: list[dict]) -> list[dict]:
        """滑动窗口 + 历史摘要，控制上下文长度"""
```

`AgentResult`:
```python
@dataclass
class AgentResult:
    success: bool
    summary: str
    steps: list[AgentStep]
    total_tokens: dict
    files_modified: list[str]
```

### Step 7: prompt_builder.py

从 `prompt_templates/` 目录读取模板文件，动态拼接。

6 种场景：

| 场景 | 模板文件 | 是否提供 tools |
|------|---------|--------------|
| judge | judge.txt | 否 |
| execute | execute.txt | **是（Agent 模式）** |
| fix | fix.txt | **是** |
| backtrack | backtrack.txt | 否 |
| integrate | integrate.txt | **是** |
| optimize | (内置) | 否 |

### Step 8: response_parser.py

只解析 **judge** 和 **backtrack** 阶段的返回（这两个阶段模型不使用 tools，返回结构化 JSON）。

execute/fix/integrate 阶段走 Agent 循环，不需要解析——模型通过 tool_call 直接操作。

### Step 9: executor.py

底层 shell 执行层，被 tools.py 调用。

```python
@dataclass
class ExecutionResult:
    command: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_ms: int
    blocked: bool          # 是否被安全过滤
    block_reason: str      # 过滤原因
```

### Step 10: processor.py — 核心递归引擎

```python
class RecursiveProcessor:
    def __init__(self, api_caller, prompt_builder, agent_loop,
                 response_parser, evaluator, persistence, config)

    async def run(self, task_description: str) -> TaskTree:
        """主入口"""
        root = TaskNode(description=task_description)
        self.tree.add_node(root)
        await self.process(root)
        return self.tree

    async def process(self, task: TaskNode) -> None:
        """
        核心递归:
        1. 调 judge prompt（无 tools），解析返回
        2. 如果 can_verify → 进入 Agent 执行循环
        3. 如果不能 → 拆分为子任务，递归处理
        """

    async def execute_with_agent(self, task: TaskNode) -> None:
        """
        用 Agent 循环执行叶子任务:
        1. 构造 execute prompt + tools
        2. 启动 agent_loop
        3. 运行验证命令
        4. 通过 → PASSED；失败 → 进入 fix_with_agent
        """

    async def fix_with_agent(self, task: TaskNode, error_info: str) -> None:
        """
        用 Agent 循环修复:
        1. 构造 fix prompt（包含错误信息）+ tools
        2. 启动 agent_loop
        3. 重新验证
        """

    async def backtrack(self, task: TaskNode) -> None:
        """回溯（不用 Agent，只调一次 API 获取新拆分方案）"""

    async def integration_verify(self, task: TaskNode) -> None:
        """用 Agent 循环执行集成验证"""
```

### Step 11: evaluator.py

在 v1 基础上增加：
- Agent 相关指标收集
- 迭代对比能力
- timeline 事件中包含 tool_call 详情

### Step 12: optimizer.py ★★

```python
class Optimizer:
    def __init__(self, api_caller, config)

    async def analyze(self, report: dict, prompt_templates: dict, config: dict,
                      previous_iteration: Optional[dict] = None) -> dict:
        """
        输入当前评估报告 + prompt模板 + 配置 + 上轮记录
        调用 LLM 分析，输出优化建议
        """

    def apply_suggestions(self, suggestions: dict) -> None:
        """将优化建议应用到 prompt_templates/ 和 config.yaml"""

    def save_iteration(self, iteration_data: dict) -> None:
        """保存优化历史"""

    def load_history(self) -> list[dict]:
        """加载所有历史迭代记录"""

    def print_comparison(self, before: dict, after: dict) -> str:
        """生成两次运行的对比摘要"""
```

### Step 13: cli.py

```bash
# 子命令
python -m recursive_coder run "任务描述"        # 单次运行
python -m recursive_coder iterate "任务描述"    # 迭代优化
python -m recursive_coder history               # 查看优化历史
python -m recursive_coder compare <id1> <id2>   # 对比两次运行
python -m recursive_coder report <run_id>       # 查看评估报告
python -m recursive_coder resume <run_id>       # 从中断恢复
```

---

## 六、实施顺序

```
阶段 A：基础设施（独立，可并行）
  A1. pyproject.toml + config.yaml + .gitignore
  A2. logger_setup.py
  A3. models.py（完善）
  A4. prompt_templates/*.txt（初版模板）

阶段 B：底层模块（有依赖，按序）
  B1. executor.py（纯 shell 执行，无 LLM 依赖）
  B2. api_caller.py（完善，支持 tool_use）
  B3. tools.py（依赖 executor）
  B4. response_parser.py
  B5. persistence.py

阶段 C：核心引擎
  C1. agent_loop.py（依赖 api_caller + tools）
  C2. prompt_builder.py（依赖 prompt_templates）
  C3. processor.py（依赖上面所有）
  C4. evaluator.py

阶段 D：迭代优化
  D1. optimizer.py
  D2. cli.py + __main__.py

阶段 E：测试与验证
  E1. 单元测试（mock，不调 API）
  E2. 集成测试 1：Hello World（用 DeepSeek API）
  E3. 集成测试 2：计算器
  E4. 首轮迭代优化实验
  E5. 集成测试 3：TODO 应用
```

---

## 七、测试方案

### 测试任务（由简到难）

| # | 任务 | 预期行为 | 评估重点 |
|---|------|---------|---------|
| 1 | `add(a,b)` 函数 + 测试 | 不拆分，直接 Agent 执行 | Agent 循环基本功能 |
| 2 | 计算器（加减乘除+括号） | 拆分 2-3 层 | 拆分质量、集成验证 |
| 3 | TODO 命令行应用 | 拆分 3+ 层 | 多模块协作、回溯 |

### 每次测试后的检查清单

- [ ] 代码是否真的能运行？
- [ ] evaluation_report.json 指标是否合理？
- [ ] API 调用日志是否完整（每个 call_XXX.json）？
- [ ] Agent 的 tool 调用链是否合理（有没有重复/无效操作）？
- [ ] 安全过滤是否正常工作？
- [ ] 任务树结构是否合理（`task_tree.json`）？

### 迭代实验设计

用 **测试任务 2（计算器）** 作为基准任务，跑 3 轮迭代优化：

```
Round 1: 默认配置 → 报告 → 优化建议
Round 2: 应用优化 → 报告 → 对比 Round 1 → 优化建议
Round 3: 应用优化 → 报告 → 对比 Round 2 → 总结
```

预期观察：
- 一次通过率逐轮提升
- parse_failures 逐轮降低
- prompt 模板逐步收敛到稳定版本
- 总成本基本持平或降低

---

## 八、风险 & 应对

| 风险 | 严重程度 | 应对 |
|------|---------|------|
| DeepSeek function calling 不稳定 | 高 | 用 `<json>` 标签做 fallback 解析；评估 parse_failures 后决定是否切换模型 |
| Agent 循环陷入无限操作 | 高 | max_agent_steps 硬限制；无效步数检测（连续 3 步无文件变更则强制停止） |
| 迭代优化建议质量差 | 中 | 默认人工审核模式；optimizer 用较强模型 |
| shell 安全问题 | 高 | 白名单模式默认开启；所有命令有完整日志可审计 |
| 上下文超出模型限制 | 中 | 滑动窗口 + 摘要压缩；监控每次调用的 token 数 |
| 接口变更连锁反应 | 中 | 接口文件作为 context_files 传递；变更时标记受影响任务 |
