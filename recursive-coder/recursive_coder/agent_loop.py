"""Agent multi-turn tool-use loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .api_caller import APICaller
from .logger_setup import get_logger
from .models import AgentStep, TaskNode, ToolCallRecord
from .tools import TOOL_DEFINITIONS, ToolExecutor

logger = get_logger("agent")


@dataclass
class AgentResult:
    success: bool = False
    summary: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    total_tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    files_modified: list[str] = field(default_factory=list)


class AgentLoop:
    """Run a multi-turn tool-use loop for a single task."""

    def __init__(
        self,
        api_caller: APICaller,
        tool_executor: ToolExecutor,
        max_steps: int = 30,
        context_window: int = 10,
        idle_detection: int = 3,
        model_name: Optional[str] = None,
    ) -> None:
        self.api = api_caller
        self.tools = tool_executor
        self.max_steps = max_steps
        self.context_window = context_window
        self.idle_detection = idle_detection
        self.model_name = model_name

    async def run(
        self,
        task: TaskNode,
        system_prompt: str,
        user_prompt: str,
    ) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        result = AgentResult()
        idle_counter = 0
        files_before: set[str] = set()

        for step_num in range(1, self.max_steps + 1):
            # compress history if needed
            messages = self._compress(messages)

            # call LLM
            resp_data = await self.api.call(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                task_node_id=task.id,
                phase="agent",
                model_name=self.model_name,
            )

            msg = resp_data["choices"][0]["message"]
            usage = resp_data.get("usage", {})
            result.total_tokens["input"] += usage.get("prompt_tokens", 0)
            result.total_tokens["output"] += usage.get("completion_tokens", 0)

            step = AgentStep(step_number=step_num, timestamp=time.time())

            # Case 1: model made tool calls
            tool_calls_raw = msg.get("tool_calls")
            if tool_calls_raw:
                # append the assistant message (with tool_calls) to history
                messages.append(msg)

                step.assistant_message = msg.get("content") or ""
                has_file_change = False

                for tc in tool_calls_raw:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError):
                        fn_args = {}

                    # Check for task_done
                    if fn_name == "task_done":
                        result.success = fn_args.get("success", False)
                        result.summary = fn_args.get("summary", "")
                        record = ToolCallRecord(
                            tool_name="task_done", arguments=fn_args,
                            result_summary=result.summary, success=result.success,
                        )
                        step.tool_calls.append(record)
                        result.steps.append(step)
                        logger.info(
                            "Agent finished at step %d: success=%s",
                            step_num, result.success,
                        )
                        # add the tool response to keep the conversation valid
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result.summary,
                        })
                        return result

                    # Execute tool
                    record = await self.tools.execute(fn_name, fn_args)
                    step.tool_calls.append(record)

                    if fn_name == "write_file" and record.success:
                        has_file_change = True
                        result.files_modified.append(fn_args.get("path", ""))

                    # append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": record.result_summary,
                    })

                # idle detection
                if has_file_change:
                    idle_counter = 0
                else:
                    idle_counter += 1

            # Case 2: pure text response (no tool calls)
            else:
                content = msg.get("content", "")
                step.assistant_message = content
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "请继续执行，使用工具完成任务。完成后调用 task_done。",
                })
                idle_counter += 1

            result.steps.append(step)

            # idle detection: if N consecutive steps without file changes, stop
            if idle_counter >= self.idle_detection:
                logger.warning(
                    "Agent idle for %d steps, forcing stop at step %d",
                    self.idle_detection, step_num,
                )
                result.summary = f"Agent stopped: idle for {self.idle_detection} steps"
                return result

        # exceeded max steps
        logger.warning("Agent exceeded max_steps=%d", self.max_steps)
        result.summary = f"Agent stopped: exceeded {self.max_steps} steps"
        return result

    def _compress(self, messages: list[dict]) -> list[dict]:
        """Keep system + task prompt + recent N turns. Summarize the rest."""
        # Each "turn" is roughly 2 messages (assistant + tool/user).
        # We keep first 2 messages (system, user) and the most recent ones.
        max_msgs = 2 + self.context_window * 3  # ~3 messages per tool-use turn
        if len(messages) <= max_msgs:
            return messages

        system = messages[0]
        task_prompt = messages[1]
        recent = messages[-max_msgs + 2 :]

        # Build a brief summary of what was dropped
        dropped = messages[2 : -max_msgs + 2]
        tool_names = []
        for m in dropped:
            if isinstance(m.get("content"), str) and m.get("role") == "tool":
                continue
            tc = m.get("tool_calls")
            if tc:
                for t in tc:
                    tool_names.append(t.get("function", {}).get("name", "?"))

        summary = f"[Earlier: {len(dropped)} messages, tools used: {', '.join(tool_names[:20])}]"
        return [
            system,
            task_prompt,
            {"role": "user", "content": summary},
            *recent,
        ]
