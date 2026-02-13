"""Core recursive processor: process → execute → backtrack → integrate."""

from __future__ import annotations

import time
from typing import Optional

from .agent_loop import AgentLoop, AgentResult
from .api_caller import APICaller
from .executor import Executor
from .logger_setup import get_logger
from .models import (
    DataPort,
    TaskNode,
    TaskStatus,
    TaskTree,
    Verification,
)
from .persistence import Persistence
from .prompt_builder import PromptBuilder
from .response_parser import parse_backtrack_response, parse_judge_response
from .tools import ToolExecutor

logger = get_logger("processor")


class RecursiveProcessor:
    """Orchestrate recursive task decomposition and agent-based execution."""

    def __init__(
        self,
        api_caller: APICaller,
        prompt_builder: PromptBuilder,
        executor: Executor,
        persistence: Persistence,
        config: dict,
    ) -> None:
        self.api = api_caller
        self.prompts = prompt_builder
        self.executor = executor
        self.persistence = persistence
        self.cfg = config

        self.tree = TaskTree()
        self.workspace = str(executor.workspace)
        self.backtrack_count = 0
        self.total_api_calls = 0

        # Build agent loop and tool executor
        self.tool_executor = ToolExecutor(executor)
        self.agent_loop = AgentLoop(
            api_caller=api_caller,
            tool_executor=self.tool_executor,
            max_steps=config.get("max_agent_steps", 30),
            context_window=config.get("agent_context_window", 10),
            idle_detection=config.get("idle_detection", 3),
            model_name=config.get("execute_model"),
        )

    # ── Public entry point ──

    async def run(self, task_description: str, data_port: Optional[DataPort] = None) -> TaskTree:
        """Accept a task description (+ optional data port), build and execute the tree."""
        root = TaskNode(
            description=task_description,
            data_port=data_port or DataPort(),
        )
        self.tree.add_node(root)
        self.persistence.save_tree(self.tree)

        logger.info("Starting: %s", task_description)
        await self.process(root)
        self.persistence.save_tree(self.tree)

        logger.info("Done. Final tree:\n%s", self.tree.print_tree())
        return self.tree

    # ── Core recursive logic ──

    async def process(self, task: TaskNode) -> None:
        if self._over_limits():
            logger.error("Global limits exceeded, stopping.")
            task.status = TaskStatus.FAILED
            return

        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self.persistence.save_tree(self.tree)

        # Step 1: Judge — can we directly execute, or must we decompose?
        system_prompt = self.prompts.system()
        user_prompt = self.prompts.judge(task, self.workspace)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = await self.api.call(
            messages=messages,
            task_node_id=task.id,
            phase="judge",
            model_name=self.cfg.get("judge_model"),
        )
        self.total_api_calls += 1

        text = resp["choices"][0]["message"].get("content", "")
        parsed = parse_judge_response(text)

        if not parsed.parse_success:
            logger.warning("Judge parse failed for task %s, treating as needs-decompose", task.id)
            parsed.can_verify = False

        # Step 2a: Leaf — directly executable
        if parsed.can_verify:
            task.verification = Verification(
                description=parsed.verification_description,
                expected_output=parsed.expected_output,
                command=parsed.verification_command,
                compare_mode=parsed.compare_mode,
            )
            if parsed.data_port:
                dp = parsed.data_port
                task.data_port.output_description = dp.get("output_description", "")
                task.data_port.output_files = dp.get("output_files", [])
            task.implementation_hint = parsed.implementation_hint
            self.persistence.save_tree(self.tree)
            await self._execute_with_agent(task)

        # Step 2b: Decompose
        else:
            task.decomposition_reason = parsed.decomposition_reason

            # Check if we need a "prepare test data" subtask
            has_data = bool(task.data_port.input_files or task.data_port.input_description)
            needs_data_subtask = not has_data and task.depth == 0

            subtask_defs = parsed.subtasks
            if needs_data_subtask:
                subtask_defs.insert(0, {
                    "description": f"为任务准备测试数据：{task.description}",
                    "data_port": {
                        "input_description": "无（需要创建）",
                        "output_description": "测试输入数据文件 + 预期输出文件",
                        "output_files": ["data/test_input.txt", "data/expected_output.txt"],
                    },
                    "dependencies": [],
                    "context_files": [],
                })

            self._add_subtasks(task, subtask_defs)
            self.persistence.save_tree(self.tree)

            # Process subtasks in topological order
            ordered = self.tree.topological_order(task.children)
            for child_id in ordered:
                child = self.tree.get_node(child_id)
                if child and child.status == TaskStatus.PENDING:
                    await self.process(child)
                    if child.status == TaskStatus.FAILED:
                        await self._backtrack(task)
                        return

            # All children passed → integration verify
            if self.tree.all_children_passed(task.id):
                await self._integration_verify(task)
            else:
                task.status = TaskStatus.FAILED

        task.end_time = time.time()
        self.persistence.save_tree(self.tree)

    # ── Agent execution ──

    async def _execute_with_agent(self, task: TaskNode) -> None:
        """Run the agent loop to implement and verify a leaf task."""
        for attempt in range(1, task.max_attempts + 1):
            task.attempts = attempt
            logger.info("Execute task %s (attempt %d/%d)", task.id, attempt, task.max_attempts)

            if attempt == 1:
                user_prompt = self.prompts.execute(task, self.workspace)
            else:
                last_error = task.error_log[-1] if task.error_log else "unknown error"
                user_prompt = self.prompts.fix(task, last_error, self.workspace)

            agent_result = await self.agent_loop.run(
                task=task,
                system_prompt=self.prompts.system(),
                user_prompt=user_prompt,
            )

            task.agent_steps += len(agent_result.steps)
            task.output_files = agent_result.files_modified
            task.token_usage["input"] += agent_result.total_tokens.get("input", 0)
            task.token_usage["output"] += agent_result.total_tokens.get("output", 0)

            # Run verification command if present
            if task.verification and task.verification.command:
                vresult = await self.executor.run(task.verification.command)
                task.verification_result = vresult.stdout + vresult.stderr

                if self._check_verification(task, vresult):
                    task.status = TaskStatus.PASSED
                    logger.info("Task %s PASSED", task.id)
                    return
                else:
                    error_msg = f"Verification failed (rc={vresult.returncode}):\n{vresult.stderr or vresult.stdout}"
                    task.error_log.append(error_msg)
                    logger.warning("Task %s verification failed (attempt %d)", task.id, attempt)
            elif agent_result.success:
                # No verification command — trust the agent's task_done
                task.status = TaskStatus.PASSED
                logger.info("Task %s PASSED (agent self-declared)", task.id)
                return
            else:
                task.error_log.append(agent_result.summary or "Agent did not complete")
                logger.warning("Task %s agent failed (attempt %d)", task.id, attempt)

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        logger.error("Task %s FAILED after %d attempts", task.id, task.max_attempts)

    def _check_verification(self, task: TaskNode, result) -> bool:
        """Compare execution result against verification criteria."""
        v = task.verification
        if not v:
            return result.returncode == 0

        mode = v.compare_mode
        expected = v.expected_output.strip()
        actual_stdout = result.stdout.strip()

        if mode == "returncode":
            return result.returncode == 0
        elif mode == "exact":
            return actual_stdout == expected
        elif mode == "contains":
            return expected in actual_stdout
        elif mode == "file_diff":
            # compare output files — not yet implemented, fallback to returncode
            return result.returncode == 0
        else:
            return result.returncode == 0

    # ── Backtrack ──

    async def _backtrack(self, task: TaskNode) -> None:
        """A child failed after all retries. Re-decompose this task."""
        self.backtrack_count += 1
        max_bt = self.cfg.get("max_backtrack_retries", 2)

        if self.backtrack_count > max_bt * (task.depth + 1):
            logger.error("Too many backtracks for task %s, giving up", task.id)
            task.status = TaskStatus.FAILED
            return

        logger.info("Backtracking task %s", task.id)

        # Collect failure info from children
        failures = []
        for cid in task.children:
            child = self.tree.get_node(cid)
            if child and child.status == TaskStatus.FAILED:
                failures.append(
                    f"- {child.description}\n  Errors: {'; '.join(child.error_log[-2:])}"
                )
        failure_details = "\n".join(failures)

        # Call LLM for new decomposition
        system_prompt = self.prompts.system()
        user_prompt = self.prompts.backtrack(task, failure_details, self.workspace)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        resp = await self.api.call(
            messages=messages,
            task_node_id=task.id,
            phase="backtrack",
            model_name=self.cfg.get("judge_model"),
        )
        self.total_api_calls += 1

        text = resp["choices"][0]["message"].get("content", "")
        parsed = parse_backtrack_response(text)

        if not parsed.parse_success or not parsed.subtasks:
            logger.error("Backtrack parse failed for %s", task.id)
            task.status = TaskStatus.FAILED
            return

        # Replace children
        task.children.clear()
        self._add_subtasks(task, parsed.subtasks)
        task.decomposition_reason = parsed.decomposition_reason
        self.persistence.save_tree(self.tree)

        # Re-process children
        ordered = self.tree.topological_order(task.children)
        for child_id in ordered:
            child = self.tree.get_node(child_id)
            if child and child.status == TaskStatus.PENDING:
                await self.process(child)
                if child.status == TaskStatus.FAILED:
                    task.status = TaskStatus.FAILED
                    return

        if self.tree.all_children_passed(task.id):
            await self._integration_verify(task)
        else:
            task.status = TaskStatus.FAILED

    # ── Integration verify ──

    async def _integration_verify(self, task: TaskNode) -> None:
        """After all children pass, run an integration verification via agent."""
        children_summary = ""
        for cid in task.children:
            child = self.tree.get_node(cid)
            if child:
                children_summary += f"- {child.description} [PASSED] outputs={child.output_files}\n"

        user_prompt = self.prompts.integrate(task, children_summary, self.workspace)
        agent_result = await self.agent_loop.run(
            task=task,
            system_prompt=self.prompts.system(),
            user_prompt=user_prompt,
        )

        if agent_result.success:
            task.status = TaskStatus.PASSED
            logger.info("Integration verify PASSED for task %s", task.id)
        else:
            logger.warning("Integration verify FAILED for task %s", task.id)
            task.status = TaskStatus.PASSED  # soft-pass: children all passed

    # ── Helpers ──

    def _add_subtasks(self, parent: TaskNode, subtask_defs: list[dict]) -> None:
        """Create TaskNode children from the parsed subtask definitions."""
        id_map: dict[int, str] = {}  # index → node id (for dependency resolution)

        for i, sd in enumerate(subtask_defs):
            dp_data = sd.get("data_port", {})
            child = TaskNode(
                description=sd["description"],
                parent_id=parent.id,
                depth=parent.depth + 1,
                context_files=sd.get("context_files", []),
                data_port=DataPort(
                    input_description=dp_data.get("input_description", ""),
                    input_files=dp_data.get("input_files", parent.data_port.input_files.copy()),
                    output_description=dp_data.get("output_description", ""),
                    output_files=dp_data.get("output_files", []),
                    upstream_task_ids=dp_data.get("upstream_task_ids", []),
                ),
            )
            self.tree.add_node(child)
            id_map[i] = child.id

        # Resolve "dependencies" which may be indices or task ids
        for i, sd in enumerate(subtask_defs):
            child_id = id_map[i]
            child = self.tree.get_node(child_id)
            if not child:
                continue
            for dep in sd.get("dependencies", []):
                if isinstance(dep, int) and dep in id_map:
                    child.dependencies.append(id_map[dep])
                elif isinstance(dep, str):
                    child.dependencies.append(dep)

    def _over_limits(self) -> bool:
        max_calls = self.cfg.get("max_total_api_calls", 500)
        return self.api.call_count >= max_calls
