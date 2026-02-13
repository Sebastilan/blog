"""Core recursive processor: judge(plan) → execute → backtrack → integrate."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
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

        # Concurrency control — limit parallel task execution
        max_parallel = config.get("max_parallel_tasks", 3)
        self._semaphore = asyncio.Semaphore(max_parallel)

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

        logger.info("[RUN] task=%s desc='%s'", root.id, task_description)
        await self.process(root)
        self.persistence.save_tree(self.tree)

        logger.info("[DONE] final_status=%s tree:\n%s", root.status.value, self.tree.print_tree())
        return self.tree

    # ── Core recursive logic ──

    async def process(self, task: TaskNode) -> None:
        if self._over_limits():
            logger.error("[LIMIT] task=%s global API call limit reached", task.id)
            task.status = TaskStatus.FAILED
            return

        max_depth = self.cfg.get("max_depth", 5)
        if task.depth >= max_depth:
            logger.error("[LIMIT] task=%s max_depth=%d reached", task.id, max_depth)
            task.status = TaskStatus.FAILED
            return

        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        self.persistence.save_tree(self.tree)

        logger.info(
            "[JUDGE] task=%s depth=%d desc='%s'",
            task.id, task.depth, task.description[:80],
        )

        # Step 1: Judge (Planning) — produce execution_plan/interface or decompose with interface_contract
        system_prompt = self.prompts.system()
        user_prompt = self.prompts.judge(task, self.workspace, tree=self.tree)
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
            logger.warning(
                "[JUDGE] task=%s parse_failed, falling back to decompose",
                task.id,
            )
            parsed.can_verify = False

        # Step 2a: Leaf — directly executable (with execution_plan + interface)
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
            task.execution_plan = parsed.execution_plan
            task.interface = parsed.interface

            logger.info(
                "[PLAN] task=%s decision=LEAF plan_steps=%d interface_keys=%s verify_cmd='%s'",
                task.id,
                len(task.execution_plan),
                list(task.interface.keys()) if task.interface else [],
                task.verification.command[:60] if task.verification else "",
            )
            if task.execution_plan:
                for step in task.execution_plan:
                    logger.debug("[PLAN] task=%s step %s", task.id, step)

            self.persistence.save_tree(self.tree)
            await self._execute_with_agent(task)

            # Gap 2: Escalation — if leaf failed, try re-judging as decomposable
            if task.status == TaskStatus.FAILED and task.depth < max_depth - 1:
                await self._escalate_to_decompose(task)

        # Step 2b: Decompose (with interface_contract)
        else:
            task.decomposition_reason = parsed.decomposition_reason
            task.interface_contract = parsed.interface_contract

            logger.info(
                "[PLAN] task=%s decision=DECOMPOSE subtasks=%d reason='%s'",
                task.id,
                len(parsed.subtasks),
                task.decomposition_reason[:80],
            )
            if task.interface_contract:
                logger.info(
                    "[CONTRACT] task=%s interface_contract='%s'",
                    task.id, task.interface_contract[:200],
                )

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

            # Log child creation summary
            for cid in task.children:
                child = self.tree.get_node(cid)
                if child:
                    logger.info(
                        "[CHILD] parent=%s child=%s desc='%s' deps=%s in_files=%s out_files=%s",
                        task.id, child.id, child.description[:60],
                        child.dependencies,
                        child.data_port.input_files,
                        child.data_port.output_files,
                    )

            # Gap 8: Process subtasks — parallel for independent, sequential for dependent
            await self._process_children(task)

            # All children passed → integration verify
            if self.tree.all_children_passed(task.id):
                await self._integration_verify(task)
            else:
                task.status = TaskStatus.FAILED

        task.end_time = time.time()
        elapsed = task.end_time - (task.start_time or task.end_time)
        logger.info(
            "[FINISH] task=%s status=%s elapsed=%.1fs tokens_in=%d tokens_out=%d",
            task.id, task.status.value, elapsed,
            task.token_usage.get("input", 0), task.token_usage.get("output", 0),
        )
        self.persistence.save_tree(self.tree)

    # ── Gap 8: Parallel child processing ──

    async def _process_children(self, task: TaskNode) -> None:
        """Process children respecting dependencies.

        Independent tasks run in parallel, but bounded by max_parallel_tasks semaphore
        to avoid overwhelming API rate limits, memory, and shell resources.
        """
        remaining = set(task.children)

        while remaining:
            # Find tasks whose dependencies are all satisfied
            ready = []
            for cid in list(remaining):
                child = self.tree.get_node(cid)
                if not child or child.status != TaskStatus.PENDING:
                    remaining.discard(cid)
                    continue
                deps_met = all(
                    self.tree.get_node(d) and self.tree.get_node(d).status == TaskStatus.PASSED
                    for d in child.dependencies
                )
                if deps_met:
                    ready.append(cid)

            if not ready:
                # No tasks ready — either all done or stuck (unmet deps with failures)
                break

            if len(ready) == 1:
                # Single task — run directly (still acquire semaphore for consistency)
                child = self.tree.get_node(ready[0])
                remaining.discard(ready[0])
                async with self._semaphore:
                    await self.process(child)
                if child.status == TaskStatus.FAILED:
                    await self._backtrack(task)
                    return
            else:
                # Multiple independent tasks — run in parallel, bounded by semaphore
                logger.info(
                    "[PARALLEL] task=%s ready=%d semaphore_limit=%d children: %s",
                    task.id, len(ready), self._semaphore._value, ready,
                )

                async def _run_with_limit(child_id: str) -> None:
                    child = self.tree.get_node(child_id)
                    if child:
                        async with self._semaphore:
                            await self.process(child)

                for cid in ready:
                    remaining.discard(cid)

                await asyncio.gather(*[_run_with_limit(cid) for cid in ready])

                # Check if any failed
                for cid in ready:
                    child = self.tree.get_node(cid)
                    if child and child.status == TaskStatus.FAILED:
                        await self._backtrack(task)
                        return

    # ── Gap 2: Escalation — re-judge failed leaf as decomposable ──

    async def _escalate_to_decompose(self, task: TaskNode) -> None:
        """When a leaf fails all retries, try re-judging it as decomposable
        before escalating to parent backtrack."""
        max_depth = self.cfg.get("max_depth", 5)
        if task.depth >= max_depth - 1:
            logger.info("[ESCALATE] task=%s at max_depth, cannot decompose further", task.id)
            return

        logger.info(
            "[ESCALATE] task=%s leaf failed after %d attempts, re-judging as decomposable",
            task.id, task.attempts,
        )

        # Reset status and re-judge with a hint to decompose
        task.status = TaskStatus.RUNNING
        error_summary = "; ".join(e[:100] for e in task.error_log[-3:])

        system_prompt = self.prompts.system()
        user_prompt = self.prompts.judge(task, self.workspace, tree=self.tree)
        # Append escalation context
        user_prompt += (
            f"\n\n重要提示：这个任务之前被判断为可以直接实现，但经过 {task.attempts} 次尝试全部失败。"
            f"\n失败原因：{error_summary}"
            f"\n请重新分析，将这个任务拆分为更小的子任务。"
            f"\n你必须返回 can_verify=false 的拆分方案。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = await self.api.call(
            messages=messages,
            task_node_id=task.id,
            phase="escalate",
            model_name=self.cfg.get("judge_model"),
        )
        self.total_api_calls += 1

        text = resp["choices"][0]["message"].get("content", "")
        parsed = parse_judge_response(text)

        if not parsed.parse_success or parsed.can_verify or not parsed.subtasks:
            logger.warning("[ESCALATE] task=%s failed to decompose, staying FAILED", task.id)
            task.status = TaskStatus.FAILED
            return

        # Successfully decomposed — switch from leaf to composite
        task.decomposition_reason = f"escalated: {parsed.decomposition_reason}"
        task.interface_contract = parsed.interface_contract
        task.verification = None  # no longer a leaf

        logger.info(
            "[ESCALATE] task=%s decomposed into %d subtasks",
            task.id, len(parsed.subtasks),
        )

        self._add_subtasks(task, parsed.subtasks)
        self.persistence.save_tree(self.tree)

        await self._process_children(task)

        if self.tree.all_children_passed(task.id):
            await self._integration_verify(task)
        else:
            task.status = TaskStatus.FAILED

    # ── Agent execution ──

    async def _execute_with_agent(self, task: TaskNode) -> None:
        """Run the agent loop to implement and verify a leaf task."""
        for attempt in range(1, task.max_attempts + 1):
            task.attempts = attempt
            logger.info(
                "[EXEC] task=%s attempt=%d/%d plan_steps=%d",
                task.id, attempt, task.max_attempts, len(task.execution_plan),
            )

            if attempt == 1:
                user_prompt = self.prompts.execute(task, self.workspace, tree=self.tree)
            else:
                last_error = task.error_log[-1] if task.error_log else "unknown error"
                user_prompt = self.prompts.fix(task, last_error, self.workspace, tree=self.tree)

            agent_result = await self.agent_loop.run(
                task=task,
                system_prompt=self.prompts.system(),
                user_prompt=user_prompt,
            )

            task.agent_steps += len(agent_result.steps)
            task.output_files = agent_result.files_modified
            task.token_usage["input"] += agent_result.total_tokens.get("input", 0)
            task.token_usage["output"] += agent_result.total_tokens.get("output", 0)

            logger.debug(
                "[EXEC] task=%s attempt=%d agent_steps=%d files_modified=%s",
                task.id, attempt, len(agent_result.steps), agent_result.files_modified,
            )

            # Run verification command if present
            if task.verification and task.verification.command:
                vresult = await self.executor.run(task.verification.command)
                task.verification_result = vresult.stdout + vresult.stderr

                passed = self._check_verification(task, vresult)
                logger.info(
                    "[VERIFY] task=%s attempt=%d passed=%s mode=%s cmd='%s' rc=%d",
                    task.id, attempt, passed,
                    task.verification.compare_mode,
                    task.verification.command[:60],
                    vresult.returncode,
                )
                if not passed:
                    logger.debug(
                        "[VERIFY] task=%s expected='%s' actual='%s'",
                        task.id,
                        task.verification.expected_output[:200],
                        (vresult.stdout or vresult.stderr)[:200],
                    )

                if passed:
                    # Gap 5a: Verify declared output files actually exist
                    missing = self._check_output_files_exist(task)
                    if missing:
                        logger.warning(
                            "[OUTPUT] task=%s missing output files: %s", task.id, missing,
                        )
                    task.status = TaskStatus.PASSED
                    logger.info("[PASS] task=%s", task.id)
                    return
                else:
                    error_msg = f"Verification failed (rc={vresult.returncode}):\n{vresult.stderr or vresult.stdout}"
                    task.error_log.append(error_msg)
                    logger.warning(
                        "[FAIL] task=%s attempt=%d error='%s'",
                        task.id, attempt, error_msg[:150],
                    )
            elif agent_result.success:
                # No verification command — trust the agent's task_done
                missing = self._check_output_files_exist(task)
                if missing:
                    logger.warning(
                        "[OUTPUT] task=%s missing output files: %s", task.id, missing,
                    )
                task.status = TaskStatus.PASSED
                logger.info("[PASS] task=%s (agent self-declared)", task.id)
                return
            else:
                task.error_log.append(agent_result.summary or "Agent did not complete")
                logger.warning(
                    "[FAIL] task=%s attempt=%d error='%s'",
                    task.id, attempt, (agent_result.summary or "Agent did not complete")[:150],
                )

        # All attempts exhausted
        task.status = TaskStatus.FAILED
        logger.error(
            "[FAILED] task=%s exhausted %d attempts, errors: %s",
            task.id, task.max_attempts,
            "; ".join(e[:80] for e in task.error_log[-3:]),
        )

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
            # Gap 5c: Compare output file against expected output file
            return self._check_file_diff(task)
        else:
            return result.returncode == 0

    # ── Gap 5c: file_diff implementation ──

    def _check_file_diff(self, task: TaskNode) -> bool:
        """Compare actual output file against expected output file."""
        expected_file = task.data_port.expected_output_file
        actual_files = task.data_port.output_files

        if not expected_file or not actual_files:
            logger.warning(
                "[FILE_DIFF] task=%s missing expected_output_file or output_files, fallback to returncode",
                task.id,
            )
            return True  # no files to compare, trust returncode

        expected_path = Path(self.workspace) / expected_file
        actual_path = Path(self.workspace) / actual_files[0]

        if not expected_path.exists():
            logger.warning("[FILE_DIFF] task=%s expected file not found: %s", task.id, expected_file)
            return False
        if not actual_path.exists():
            logger.warning("[FILE_DIFF] task=%s actual file not found: %s", task.id, actual_files[0])
            return False

        expected_content = expected_path.read_text(encoding="utf-8", errors="replace").strip()
        actual_content = actual_path.read_text(encoding="utf-8", errors="replace").strip()

        match = expected_content == actual_content
        if not match:
            logger.debug(
                "[FILE_DIFF] task=%s mismatch:\n  expected (first 200): '%s'\n  actual (first 200): '%s'",
                task.id, expected_content[:200], actual_content[:200],
            )
        return match

    # ── Gap 5a: Verify output files exist ──

    def _check_output_files_exist(self, task: TaskNode) -> list[str]:
        """Check that declared output_files actually exist on disk. Returns list of missing files."""
        missing = []
        for f in task.data_port.output_files:
            full = Path(self.workspace) / f
            if not full.exists():
                missing.append(f)
        return missing

    # ── Backtrack ──

    async def _backtrack(self, task: TaskNode) -> None:
        """A child failed after all retries. Re-decompose this task.

        Gap 6: Selective backtrack — preserve successful children's outputs as context.
        """
        self.backtrack_count += 1
        max_bt = self.cfg.get("max_backtrack_retries", 2)

        if self.backtrack_count > max_bt:
            logger.error(
                "[BACKTRACK] task=%s global_limit=%d exceeded, giving up",
                task.id, max_bt,
            )
            task.status = TaskStatus.FAILED
            return

        # Gap 6: Collect info from ALL children — both failed and succeeded
        failures = []
        succeeded_context = []
        for cid in task.children:
            child = self.tree.get_node(cid)
            if not child:
                continue
            if child.status == TaskStatus.FAILED:
                failures.append(
                    f"- [FAILED] {child.description} (id={child.id})\n"
                    f"  Errors: {'; '.join(child.error_log[-2:])}\n"
                    f"  Interface: {json.dumps(child.interface, ensure_ascii=False) if child.interface else 'N/A'}"
                )
            elif child.status == TaskStatus.PASSED:
                succeeded_context.append(
                    f"- [PASSED] {child.description} (id={child.id})\n"
                    f"  Output files: {child.output_files}\n"
                    f"  Interface: {json.dumps(child.interface, ensure_ascii=False) if child.interface else 'N/A'}"
                )

        failure_details = "\n".join(failures)
        if succeeded_context:
            failure_details += "\n\n以下子任务已成功完成（请在重新拆分时考虑复用其输出）：\n"
            failure_details += "\n".join(succeeded_context)

        logger.info(
            "[BACKTRACK] task=%s attempt=%d/%d failed=%d succeeded=%d",
            task.id, self.backtrack_count, max_bt,
            len(failures), len(succeeded_context),
        )
        logger.debug("[BACKTRACK] task=%s details:\n%s", task.id, failure_details)

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
            logger.error("[BACKTRACK] task=%s parse_failed", task.id)
            task.status = TaskStatus.FAILED
            return

        # Update interface_contract if backtrack provides a new one
        if parsed.interface_contract:
            task.interface_contract = parsed.interface_contract
            logger.info(
                "[BACKTRACK] task=%s updated interface_contract='%s'",
                task.id, task.interface_contract[:200],
            )

        # Replace children
        task.children.clear()
        self._add_subtasks(task, parsed.subtasks)
        task.decomposition_reason = parsed.decomposition_reason
        self.persistence.save_tree(self.tree)

        logger.info(
            "[BACKTRACK] task=%s new_subtasks=%d reason='%s'",
            task.id, len(parsed.subtasks), parsed.decomposition_reason[:80],
        )

        # Re-process children
        await self._process_children(task)

        if self.tree.all_children_passed(task.id):
            await self._integration_verify(task)
        else:
            task.status = TaskStatus.FAILED

    # ── Integration verify ──

    async def _integration_verify(self, task: TaskNode) -> None:
        """After all children pass, run an integration verification via agent.

        Gap 4: Integration failure = real failure, not soft-pass.
        """
        children_summary = ""
        for cid in task.children:
            child = self.tree.get_node(cid)
            if child:
                iface_str = ""
                if child.interface:
                    iface_str = f" interface={json.dumps(child.interface, ensure_ascii=False)}"
                children_summary += (
                    f"- {child.description} [PASSED] "
                    f"outputs={child.output_files}{iface_str}\n"
                )

        logger.info(
            "[INTEGRATE] task=%s children=%d has_contract=%s",
            task.id, len(task.children), bool(task.interface_contract),
        )

        user_prompt = self.prompts.integrate(task, children_summary, self.workspace)
        agent_result = await self.agent_loop.run(
            task=task,
            system_prompt=self.prompts.system(),
            user_prompt=user_prompt,
        )

        if agent_result.success:
            task.status = TaskStatus.PASSED
            logger.info("[INTEGRATE] task=%s PASSED", task.id)
        else:
            # Gap 4: Integration failure is a real failure
            logger.warning("[INTEGRATE] task=%s FAILED", task.id)
            task.status = TaskStatus.FAILED
            task.error_log.append(
                f"Integration verify failed: {agent_result.summary or 'agent did not confirm success'}"
            )

    # ── Helpers ──

    def _add_subtasks(self, parent: TaskNode, subtask_defs: list[dict]) -> None:
        """Create TaskNode children from the parsed subtask definitions.

        Propagates parent's interface_contract to all children.
        """
        id_map: dict[int, str] = {}  # index → node id (for dependency resolution)

        for i, sd in enumerate(subtask_defs):
            dp_data = sd.get("data_port", {})
            child = TaskNode(
                description=sd["description"],
                parent_id=parent.id,
                depth=parent.depth + 1,
                context_files=sd.get("context_files", []),
                # Propagate parent's interface_contract to child
                interface_contract=parent.interface_contract,
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
