# Recursive Coder - 项目备忘

## 当前状态

项目已开发完成，55 个单元测试全部通过，可以本地用 API 跑了。

## 快速开始

1. 设置 API Key（默认模型 deepseek-v3）：

```bash
export DEEPSEEK_API_KEY="sk-..."
```

其他模型对应的环境变量：
- Claude: `ANTHROPIC_API_KEY`
- GPT: `OPENAI_API_KEY`
- Gemini: `GOOGLE_API_KEY`
- Qwen: `DASHSCOPE_API_KEY`

2. 运行：

```bash
cd recursive-coder
python -m recursive_coder --project-dir . run "任务描述" --verbose
```

3. 切换模型：`--model claude-sonnet` / `gpt-4.1-mini` / `gemini-2.5-flash` / `qwen3-coder`

## 常用命令

```bash
# 单次运行
python -m recursive_coder run "任务描述" --verbose

# 迭代优化（多轮自动调参）
python -m recursive_coder iterate "任务描述" --rounds 3 --auto

# 查看报告
python -m recursive_coder report ./workspace/<run_id>

# 查看优化历史
python -m recursive_coder history

# 运行测试
python -m pytest tests/ -q

# 运行 eval（mock 模式，不消耗 API）
python eval/run_eval.py --mock
```

## 架构概览

递归任务分解框架，核心循环：judge -> execute -> verify -> (fix/backtrack) -> integrate

- `processor.py` — 递归主控（judge/execute/backtrack/integrate）
- `agent_loop.py` — 多步 agent 循环（tool-use）
- `api_caller.py` — 统一 LLM API 调用（支持 Anthropic / OpenAI / OpenAI-compatible）
- `prompt_builder.py` — 模板加载 + 上下文注入
- `models.py` — TaskNode / TaskTree 数据模型
- `executor.py` — 沙盒命令执行
- `evaluator.py` — 评估报告生成
- `optimizer.py` — 迭代优化（分析报告 -> 调整 prompt/config）

## 配置

主配置在 `config.yaml`，关键参数：
- `max_depth: 5` — 递归最大深度
- `max_retries: 3` — 单任务最大重试
- `max_total_api_calls: 500` — API 调用总量上限
- `command_timeout: 60` — 命令执行超时（秒）
