# 递归任务分解 AI 编码框架 — 实施计划

## 〇、当前状态

已完成的代码（需要修改/完善）：

| 文件 | 状态 | 说明 |
|------|------|------|
| `recursive_coder/models.py` | 基本完成 | TaskNode / TaskTree 数据结构，需增加评估相关字段 |
| `recursive_coder/api_caller.py` | 基本完成 | 多模型 API 调用，需增加每次调用的详细日志记录 |
| `recursive_coder/__init__.py` | 完成 | 版本号 |

## 一、项目文件结构（最终目标）

```
recursive-coder/
├── pyproject.toml                    # 项目依赖和元数据
├── config.yaml                       # 默认配置（模型选择、重试次数、深度上限等）
├── PLAN.md                           # 本计划文件
│
├── recursive_coder/
│   ├── __init__.py
│   ├── models.py                     # 核心数据结构（TaskNode, TaskTree, Verification）
│   ├── api_caller.py                 # LLM API 统一调用层
│   ├── prompt_builder.py             # Prompt 构造器（判断/拆分/执行/修复/回溯）
│   ├── executor.py                   # Shell 执行器（编译、运行验证命令）
│   ├── processor.py                  # 核心递归处理器（process/execute/backtrack）
│   ├── response_parser.py            # 模型返回内容的结构化解析
│   ├── persistence.py                # 任务树状态持久化（JSON 文件）
│   ├── evaluator.py                  # ★ 评估器：收集指标、生成评估报告
│   ├── logger_setup.py               # ★ 日志系统配置（结构化日志）
│   └── cli.py                        # CLI 入口
│
├── workspace/                        # 运行时生成的工作区（gitignore）
│   └── <task_id>/                    # 每次运行的独立工作区
│       ├── task_tree.json            # 任务树快照
│       ├── run.log                   # 运行日志
│       ├── evaluation_report.json    # ★ 评估报告
│       ├── api_calls/                # ★ 每次 API 调用的完整记录
│       │   ├── call_001.json         #   (prompt, response, tokens, 耗时)
│       │   ├── call_002.json
│       │   └── ...
│       └── output/                   # 生成的代码产物
│           └── ...
│
└── tests/                            # 框架自身的测试
    ├── test_models.py
    ├── test_prompt_builder.py
    ├── test_response_parser.py
    └── test_executor.py
```

## 二、实施步骤（按顺序）

### Step 1：完善 models.py — 增加评估字段

在 TaskNode 中增加以下字段，用于事后评估：

```python
# 新增字段
start_time: Optional[float] = None      # 任务开始时间
end_time: Optional[float] = None        # 任务完成时间
api_call_ids: list[str] = []            # 关联的 API 调用 ID 列表
token_usage: dict = {"input": 0, "output": 0}  # 该任务累计 token
decomposition_reason: str = ""           # 为什么拆分（模型原话摘要）
verification_result: str = ""            # 验证的 stdout/stderr
```

### Step 2：完善 api_caller.py — 增加详细调用日志

每次 API 调用生成一条完整的调用记录，写入 `workspace/<task_id>/api_calls/call_XXX.json`：

```json
{
  "call_id": "call_001",
  "timestamp": "2026-02-13T12:00:00Z",
  "model": "deepseek-v3",
  "task_node_id": "a1b2c3d4",
  "phase": "judge | execute | fix | backtrack",
  "system_prompt": "...",
  "user_prompt": "...",
  "response": "...",
  "input_tokens": 800,
  "output_tokens": 400,
  "latency_ms": 2300,
  "error": null
}
```

修改点：
- `call()` 方法增加 `task_node_id` 和 `phase` 参数
- 增加 `workspace_dir` 配置，用于写入调用记录
- 将 DeepSeek 设为默认测试模型，API key 通过环境变量 `DEEPSEEK_API_KEY` 传入

### Step 3：实现 response_parser.py — 解析模型返回

模型的返回是自然语言混合代码块的文本，需要从中提取结构化信息。

两种返回场景的解析：

**场景 A：可以构造验证用例（叶子任务）**
```
提取:
- verification.description
- verification.input_data
- verification.expected_output
- verification.command
- implementation_hint（可选）
```

**场景 B：需要继续拆分**
```
提取:
- subtasks: [{description, dependencies[], context_files[]}]
- decomposition_reason
```

**场景 C：执行阶段返回代码**
```
提取:
- 代码块（按文件名分组）
- 文件路径
```

解析策略：
1. 要求模型在返回中使用特定的标记（如 ````json ... ```）包裹结构化数据
2. 先尝试提取 JSON 块；失败则用正则匹配关键段落
3. 解析失败时记录原始返回，计为一次失败重试

### Step 4：实现 prompt_builder.py — Prompt 模板管理

5 种 prompt 场景：

| 场景 | 何时触发 | 核心指令 |
|------|---------|---------|
| **judge** | process() 入口 | "判断能否构造验证用例，能则给出，不能则拆分" |
| **execute** | 叶子任务执行 | "按验证标准完成代码，输出完整文件" |
| **fix** | 验证失败后重试 | "根据错误信息修复代码" |
| **backtrack** | 子任务多次失败 | "重新审视拆分方式，给出新方案" |
| **integrate** | 所有子任务完成后 | "编写集成验证，确认子模块协作正确" |

每个 prompt 由以下部分拼接：
1. 系统模板（固定 ~100 token）
2. 全局约定文件内容（如有）
3. 当前任务描述
4. 场景特定内容（验证标准 / 错误信息 / 子任务失败报告）
5. 相关上下文文件内容（按 context_files 读取）
6. 输出格式要求（JSON 标记块）

**关键设计：输出格式要求**

在每个 prompt 末尾附加格式指引，要求模型将结构化数据包裹在 `<json>...</json>` 标记中，便于 response_parser 提取。例如：

```
请将你的判断结果包裹在 <json> 标签中，格式如下：

如果可以构造验证用例：
<json>
{
  "can_verify": true,
  "verification": {
    "description": "...",
    "input_data": "...",
    "expected_output": "...",
    "command": "..."
  },
  "implementation_hint": "..."
}
</json>

如果需要拆分：
<json>
{
  "can_verify": false,
  "subtasks": [
    {"description": "...", "dependencies": [], "context_files": []},
    ...
  ],
  "decomposition_reason": "..."
}
</json>
```

### Step 5：实现 executor.py — Shell 执行器

职责：在隔离的工作目录中执行 shell 命令（编译、运行测试等）。

```python
class Executor:
    def __init__(self, workspace_dir: str)
    async def run(self, command: str, timeout: int = 60) -> ExecutionResult
    async def write_file(self, path: str, content: str) -> None
    async def read_file(self, path: str) -> str
```

`ExecutionResult` 包含：
- `returncode: int`
- `stdout: str`
- `stderr: str`
- `timed_out: bool`
- `duration_ms: int`

安全考虑：
- 命令执行有超时限制（默认 60 秒）
- 工作目录限制在 workspace 下
- stdout/stderr 截断上限（防止巨大输出撑爆内存），默认 10000 字符

### Step 6：实现 processor.py — 核心递归引擎 ★

这是整个系统的核心，实现设计方案中的 `process()` / `execute()` / `backtrack()` / `integration_verify()` 四个主流程。

```python
class RecursiveProcessor:
    def __init__(self, api_caller, prompt_builder, executor, response_parser, evaluator, config)

    async def run(self, task_description: str) -> TaskTree:
        """入口：接受用户原始任务描述，返回完成后的任务树"""

    async def process(self, task: TaskNode) -> None:
        """核心递归：判断 → 执行 or 拆分"""

    async def execute(self, task: TaskNode) -> None:
        """叶子任务执行：调 API 写代码 → 运行验证"""

    async def backtrack(self, task: TaskNode) -> None:
        """回溯：子任务失败时，让模型重新拆分"""

    async def integration_verify(self, task: TaskNode) -> None:
        """集成验证：所有子任务完成后验证整体"""
```

**串行 vs 并行策略：**

Phase 1 先实现串行（简单可靠）：
```
process(task):
    response = call_api(judge_prompt)
    parsed = parse(response)

    if parsed.can_verify:
        execute(task)
    else:
        for subtask in parsed.subtasks:
            tree.add_node(subtask)
        for subtask_id in tree.topological_order(task.children):
            await process(tree.get_node(subtask_id))
        integration_verify(task)
```

Phase 2 增加并行（同一层无依赖的任务并发执行）：
```
# 获取所有就绪任务（依赖已完成），用 asyncio.gather 并发
while not tree.all_children_passed(task.id):
    ready = tree.get_ready_tasks(task.children)
    await asyncio.gather(*[process(tree.get_node(r)) for r in ready])
```

**重试与回溯逻辑：**
```
execute(task):
    for attempt in range(max_attempts):
        code = call_api(execute_prompt if attempt == 0 else fix_prompt)
        write_files(code)
        result = run_verification()
        if result.success:
            task.status = PASSED; return
        task.error_log.append(result.stderr)

    task.status = FAILED
    backtrack(task.parent)

backtrack(parent):
    # 收集失败信息
    failure_info = {child: errors for failed children}
    response = call_api(backtrack_prompt with failure_info)
    # 清除旧的子任务，用新拆分替换
    replace_children(parent, new_subtasks)
    # 重新处理
    process(parent)  # 递归，但有深度/重试上限
```

**安全限制：**
- 最大递归深度：5 层（可配置）
- 单任务最大重试：3 次（可配置）
- 单次回溯最大重试：2 次（可配置）
- 总 API 调用上限：500 次（可配置，防止失控）

### Step 7：实现 persistence.py — 状态持久化

每次任务状态变化时，将完整的 TaskTree 序列化写入 `workspace/<run_id>/task_tree.json`。

用途：
1. 中断恢复：程序崩溃后可以从最后一个快照恢复
2. 事后分析：查看任务树的最终状态
3. 评估：评估器读取此文件生成报告

```python
class Persistence:
    def __init__(self, workspace_dir: str)
    def save_tree(self, tree: TaskTree) -> None
    def load_tree(self) -> Optional[TaskTree]
    def save_api_call(self, call_record: dict) -> None
```

### Step 8：实现 evaluator.py — 评估体系 ★★

这是你特别强调的部分。评估器在运行结束后（或运行中实时）收集数据、计算指标、生成报告。

#### 8.1 评估指标体系

分三个维度：**效率**、**质量**、**过程**。

**效率指标：**

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 总 API 调用次数 | 直接计数 | 衡量整体调用开销 |
| 总 token 消耗 | input + output 分别统计 | 衡量成本 |
| 预估费用（$） | token × 单价 | 直观成本 |
| 总耗时（秒） | end_time - start_time | 端到端时间 |
| API 调用延迟分布 | P50/P90/P99 | 识别慢调用 |
| 有效调用比 | 成功调用 / 总调用 | 判断调用是否浪费 |

**质量指标：**

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 一次通过率 | 首次验证通过的叶子 / 总叶子 | 模型代码质量 |
| 平均重试次数 | 所有叶子的 attempts 均值 | 重试代价 |
| 回溯次数 | backtrack 触发次数 | 拆分质量 |
| 最终通过率 | PASSED 叶子 / 总叶子 | 最终完成度 |
| 集成验证通过率 | 通过的中间节点 / 总中间节点 | 组合质量 |
| 最终产物是否可运行 | 手动/自动验证 | 最终结果 |

**过程指标：**

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| 任务树深度 | 最大深度 | 拆分是否过深 |
| 任务树宽度 | 最大单节点子任务数 | 拆分是否过宽 |
| 叶子任务数量 | 叶子节点计数 | 问题拆解粒度 |
| 解析失败次数 | response_parser 失败计数 | prompt 设计质量 |
| 按深度统计的通过率 | 各深度层的 pass/fail | 找到薄弱环节 |

#### 8.2 评估报告格式

运行结束后生成 `evaluation_report.json`：

```json
{
  "run_id": "20260213_143000",
  "task_description": "实现一个简单的计算器",
  "model": "deepseek-v3",
  "status": "completed | partial | failed",

  "efficiency": {
    "total_api_calls": 45,
    "total_input_tokens": 38000,
    "total_output_tokens": 22000,
    "estimated_cost_usd": 0.03,
    "total_duration_seconds": 180,
    "api_latency_p50_ms": 1200,
    "api_latency_p90_ms": 3500,
    "effective_call_ratio": 0.82
  },

  "quality": {
    "first_pass_rate": 0.75,
    "avg_retries": 1.3,
    "backtrack_count": 2,
    "final_pass_rate": 0.95,
    "integration_pass_rate": 1.0
  },

  "process": {
    "tree_max_depth": 3,
    "tree_max_width": 5,
    "total_leaf_tasks": 20,
    "total_intermediate_tasks": 8,
    "parse_failures": 3,
    "pass_rate_by_depth": {"0": 1.0, "1": 1.0, "2": 0.9, "3": 0.85}
  },

  "timeline": [
    {"timestamp": "...", "event": "task_created", "task_id": "...", "detail": "..."},
    {"timestamp": "...", "event": "api_call", "task_id": "...", "detail": "..."},
    ...
  ]
}
```

#### 8.3 评估器实现

```python
class Evaluator:
    def __init__(self, workspace_dir: str)

    def record_event(self, event_type: str, task_id: str, detail: dict) -> None
        """实时记录事件到 timeline"""

    def generate_report(self, tree: TaskTree) -> dict
        """运行结束后生成完整评估报告"""

    def print_summary(self, report: dict) -> str
        """生成人类可读的摘要，输出到终端"""
```

### Step 9：实现 logger_setup.py — 结构化日志

两路日志输出：

1. **控制台**：简洁的进度信息（彩色），格式：`[时间] [级别] [task_id] 消息`
2. **文件**：完整的结构化日志，写入 `workspace/<run_id>/run.log`

```python
def setup_logging(workspace_dir: str, verbose: bool = False) -> None:
    """配置 logging，同时输出到控制台和文件"""
```

日志级别策略：
- INFO：任务状态变化、API 调用摘要、验证结果
- DEBUG：完整的 prompt 和 response（仅写入文件）
- WARNING：解析失败、重试
- ERROR：任务失败、回溯触发

### Step 10：实现 cli.py — CLI 入口

```bash
# 基本用法
python -m recursive_coder "实现一个简单的计算器，支持加减乘除"

# 指定模型和工作区
python -m recursive_coder "..." --model deepseek-v3 --workspace ./my_workspace

# 从中断恢复
python -m recursive_coder --resume ./workspace/20260213_143000

# 查看历史运行的评估报告
python -m recursive_coder --report ./workspace/20260213_143000
```

CLI 参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `task` | (必填) | 任务描述 |
| `--model` | `deepseek-v3` | 默认模型 |
| `--workspace` | `./workspace` | 工作区目录 |
| `--max-depth` | `5` | 最大递归深度 |
| `--max-retries` | `3` | 单任务最大重试 |
| `--max-calls` | `500` | 总 API 调用上限 |
| `--timeout` | `60` | 单命令超时秒数 |
| `--parallel` | `false` | 是否启用并行执行 |
| `--resume` | `None` | 恢复运行的工作区路径 |
| `--report` | `None` | 查看评估报告 |
| `--verbose` | `false` | 详细日志 |

### Step 11：配置和依赖

**pyproject.toml 依赖：**
- `httpx` — 异步 HTTP 请求（调 API）
- `pyyaml` — 配置文件解析
- 仅使用标准库：`asyncio`, `json`, `logging`, `subprocess`, `argparse`, `dataclasses`, `uuid`, `time`, `pathlib`

无重型依赖，保持轻量。

**config.yaml 默认配置：**
```yaml
default_model: "deepseek-v3"
judge_model: null          # 为空则使用 default_model
execute_model: null         # 为空则使用 default_model
max_depth: 5
max_retries: 3
max_backtrack_retries: 2
max_total_calls: 500
command_timeout: 60
parallel: false
```

### Step 12：框架自身的测试

| 测试文件 | 测试内容 |
|---------|---------|
| `test_models.py` | TaskNode/TaskTree 的序列化/反序列化、拓扑排序、状态管理 |
| `test_prompt_builder.py` | 各场景 prompt 的正确拼接 |
| `test_response_parser.py` | 各种模型返回格式的解析（正常/异常/边界情况） |
| `test_executor.py` | 命令执行、超时、输出截断 |

这些测试不依赖外部 API，用 mock 数据。

## 三、测试方案

使用 **DeepSeek V3** (`DEEPSEEK_API_KEY` 环境变量) 作为测试模型。

### 测试任务（由简到难）

**测试 1：Hello World 级别**
```
"用 Python 实现一个函数 add(a, b)，返回两个数的和，并写一个测试验证它。"
```
预期：不需要拆分，直接构造验证用例并执行。验证一次通过。

**测试 2：中等复杂度**
```
"用 Python 实现一个简单的计算器，支持加减乘除和括号表达式解析，写测试验证。"
```
预期：拆分为 2-3 层，约 5-10 个叶子任务。

**测试 3：较高复杂度**
```
"用 Python 实现一个命令行 TODO 应用，支持添加/删除/列出/标记完成任务，数据持久化到 JSON 文件。"
```
预期：拆分为 3+ 层，约 10-20 个叶子任务，涉及多模块集成。

### 每次测试后评估

运行完成后检查：
1. `evaluation_report.json` 中各指标是否合理
2. 生成的代码是否真的能运行
3. API 调用日志是否完整可追溯
4. 识别问题点 → 调整 prompt 模板或流程参数

## 四、实施顺序和依赖关系

```
Step 1 (models.py 完善)  ──┐
                           ├── Step 6 (processor.py) ── Step 10 (cli.py)
Step 2 (api_caller.py)  ───┤                              │
Step 3 (response_parser) ──┤                              │
Step 4 (prompt_builder) ───┤                              ▼
Step 5 (executor.py) ──────┘                         测试运行
                                                         │
Step 7 (persistence.py) ──── Step 6 也需要              │
Step 8 (evaluator.py) ────── Step 6 也需要              │
Step 9 (logger_setup.py) ─── 所有模块都需要             │
Step 11 (config/deps) ────── 最先做                     ▼
Step 12 (tests) ───────────────────────────────── 最后做
```

**实际实施顺序：**

1. **Step 11** → pyproject.toml, config.yaml, .gitignore
2. **Step 9** → logger_setup.py（所有模块依赖它）
3. **Step 1** → 完善 models.py
4. **Step 2** → 完善 api_caller.py
5. **Step 3** → response_parser.py
6. **Step 4** → prompt_builder.py
7. **Step 5** → executor.py
8. **Step 8** → evaluator.py
9. **Step 7** → persistence.py
10. **Step 6** → processor.py（核心，依赖前面所有模块）
11. **Step 10** → cli.py + `__main__.py`
12. **Step 12** → 单元测试
13. **集成测试** → 用 DeepSeek API 跑测试任务 1/2/3

## 五、关键设计决策说明

### 为什么用 `<json>` 标签而不是纯 JSON 返回？

模型返回纯 JSON 经常出错（格式不对、多余文字）。用标签包裹的方式：
- 允许模型先用自然语言"思考"，然后输出结构化结果
- 解析更可靠（正则提取标签内容）
- 解析失败时还有自然语言部分可做 fallback

### 为什么默认串行而不是并行？

- 串行更容易调试和评估
- 并行引入的竞态条件（文件冲突、依赖管理）增加复杂度
- Phase 1 先跑通串行，评估报告确认流程正确后再开并行

### DeepSeek 作为测试模型的理由

- 成本极低（$0.27/M token），适合频繁调试
- 代码能力在开源模型中位于前列
- OpenAI 兼容 API，验证通用性
- 你提供了可用的 API key

## 六、风险 & 应对

| 风险 | 应对 |
|------|------|
| DeepSeek 返回格式不稳定，解析频繁失败 | response_parser 做多层 fallback；记录解析失败率到评估指标；据此调整 prompt |
| 递归拆分过深，叶子任务太碎 | 深度上限 5 层；evaluator 监控 tree_max_depth，过深时调整 prompt |
| 验证命令不可靠（误报通过/误报失败） | 集成验证兜底；评估报告中单独标注集成验证结果 |
| 模型生成的代码存在安全问题 | executor 在隔离工作目录运行；设命令超时；不执行 rm/网络请求等危险命令 |
| API 调用失控（成本暴涨） | max_total_calls 硬上限；evaluator 实时跟踪 token 消耗 |
